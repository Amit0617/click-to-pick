/**
 * Damped least-squares pose IK.
 *
 * This module deliberately knows nothing about MuJoCo. A kinematics provider
 * supplies the end-effector pose and a 6 x n geometric Jacobian.
 */
import * as THREE from 'three';

export interface IKPose {
    position: THREE.Vector3;
    quaternion: THREE.Quaternion;
}

export interface IKKinematics {
    evaluate(joints: readonly number[]): { pose: IKPose; jacobian: number[][] };
}

export interface DampedLeastSquaresOptions {
    maxIterations?: number;
    positionTolerance?: number;
    rotationTolerance?: number;
    positionWeight?: number;
    rotationWeight?: number;
    initialDamping?: number;
    minDamping?: number;
    maxDamping?: number;
    maxJointStep?: number;
    lowerLimits?: readonly number[];
    upperLimits?: readonly number[];
}

export interface DampedLeastSquaresResult {
    joints: number[];
    converged: boolean;
    iterations: number;
    positionError: number;
    rotationError: number;
    cost: number;
}

const DEFAULTS: Required<Omit<DampedLeastSquaresOptions, 'lowerLimits' | 'upperLimits'>> = {
    maxIterations: 80,
    positionTolerance: 1e-3,
    rotationTolerance: 1e-2,
    positionWeight: 1,
    rotationWeight: 0.5,
    initialDamping: 1e-2,
    minDamping: 1e-6,
    maxDamping: 1e3,
    maxJointStep: 0.2,
};

function solveLinearSystem(matrix: number[][], rhs: number[]): number[] | null {
    const n = rhs.length;
    const a = matrix.map((row, i) => [...row, rhs[i]]);

    // Gaussian elimination with partial pivoting. IK systems are small (6-7 DOF).
    for (let col = 0; col < n; col++) {
        let pivot = col;
        for (let row = col + 1; row < n; row++) {
            if (Math.abs(a[row][col]) > Math.abs(a[pivot][col])) pivot = row;
        }
        if (Math.abs(a[pivot][col]) < 1e-12) return null;
        [a[col], a[pivot]] = [a[pivot], a[col]];

        const divisor = a[col][col];
        for (let j = col; j <= n; j++) a[col][j] /= divisor;

        for (let row = 0; row < n; row++) {
            if (row === col) continue;
            const factor = a[row][col];
            for (let j = col; j <= n; j++) a[row][j] -= factor * a[col][j];
        }
    }
    return a.map(row => row[n]);
}

function rotationVector(error: THREE.Quaternion): THREE.Vector3 {
    const q = error.clone().normalize();
    // q and -q represent the same rotation. Choose the shortest arc.
    if (q.w < 0) q.set(-q.x, -q.y, -q.z, -q.w);
    const sinHalf = Math.hypot(q.x, q.y, q.z);
    if (sinHalf < 1e-10) return new THREE.Vector3(2 * q.x, 2 * q.y, 2 * q.z);
    const angle = 2 * Math.atan2(sinHalf, Math.max(-1, Math.min(1, q.w)));
    return new THREE.Vector3(q.x, q.y, q.z).multiplyScalar(angle / sinHalf);
}

function poseError(current: IKPose, target: IKPose): number[] {
    const position = target.position.clone().sub(current.position);
    // World-frame orientation error: q_target * inverse(q_current).
    const orientation = target.quaternion.clone().multiply(current.quaternion.clone().invert());
    const angular = rotationVector(orientation);
    return [position.x, position.y, position.z, angular.x, angular.y, angular.z];
}

function objective(error: number[], positionWeight: number, rotationWeight: number): number {
    return positionWeight * error.slice(0, 3).reduce((sum, x) => sum + x * x, 0)
        + rotationWeight * error.slice(3, 6).reduce((sum, x) => sum + x * x, 0);
}

function clampJoints(
    joints: number[],
    lower?: readonly number[],
    upper?: readonly number[],
): number[] {
    return joints.map((q, i) => Math.max(
        lower?.[i] ?? -Infinity,
        Math.min(upper?.[i] ?? Infinity, q),
    ));
}

export class DampedLeastSquaresIK {
    private readonly options: typeof DEFAULTS & Pick<DampedLeastSquaresOptions, 'lowerLimits' | 'upperLimits'>;

    constructor(private readonly kinematics: IKKinematics, options: DampedLeastSquaresOptions = {}) {
        this.options = { ...DEFAULTS, ...options };
    }

    solve(target: IKPose, initialJoints: readonly number[]): DampedLeastSquaresResult {
        const o = this.options;
        let joints = clampJoints([...initialJoints], o.lowerLimits, o.upperLimits);
        let damping = o.initialDamping;
        let iterations = 0;
        let state = this.kinematics.evaluate(joints);
        let error = poseError(state.pose, target);
        let cost = objective(error, o.positionWeight, o.rotationWeight);

        for (; iterations < o.maxIterations; iterations++) {
            const positionError = Math.hypot(error[0], error[1], error[2]);
            const rotationError = Math.hypot(error[3], error[4], error[5]);
            if (positionError <= o.positionTolerance && rotationError <= o.rotationTolerance) break;

            const n = joints.length;
            if (!n || state.jacobian.length !== 6 || state.jacobian.some(row => row.length !== n)) break;

            const weights = [o.positionWeight, o.positionWeight, o.positionWeight,
                o.rotationWeight, o.rotationWeight, o.rotationWeight];
            const normal = Array.from({ length: n }, () => Array(n).fill(0));
            const rhs = Array(n).fill(0);

            for (let r = 0; r < 6; r++) {
                for (let i = 0; i < n; i++) {
                    const ji = state.jacobian[r][i];
                    rhs[i] += weights[r] * ji * error[r];
                    for (let j = 0; j < n; j++) {
                        normal[i][j] += weights[r] * ji * state.jacobian[r][j];
                    }
                }
            }
            for (let i = 0; i < n; i++) normal[i][i] += damping * damping;

            const delta = solveLinearSystem(normal, rhs);
            if (!delta) {
                damping = Math.min(o.maxDamping, damping * 10);
                continue;
            }

            const maxAbsStep = Math.max(...delta.map(Math.abs));
            const scale = maxAbsStep > o.maxJointStep ? o.maxJointStep / maxAbsStep : 1;
            const candidate = clampJoints(
                joints.map((q, i) => q + delta[i] * scale),
                o.lowerLimits,
                o.upperLimits,
            );
            const candidateState = this.kinematics.evaluate(candidate);
            const candidateError = poseError(candidateState.pose, target);
            const candidateCost = objective(candidateError, o.positionWeight, o.rotationWeight);

            if (Number.isFinite(candidateCost) && candidateCost < cost) {
                joints = candidate;
                state = candidateState;
                error = candidateError;
                cost = candidateCost;
                damping = Math.max(o.minDamping, damping * 0.5);
            } else {
                damping = Math.min(o.maxDamping, damping * 4);
                if (damping >= o.maxDamping) break;
            }
        }

        const positionError = Math.hypot(error[0], error[1], error[2]);
        const rotationError = Math.hypot(error[3], error[4], error[5]);
        return {
            joints,
            converged: positionError <= o.positionTolerance && rotationError <= o.rotationTolerance,
            iterations,
            positionError,
            rotationError,
            cost,
        };
    }
}
