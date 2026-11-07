/**
 * MuJoCo implementation of the IKKinematics contract.
 *
 * Uses an isolated MjData so numerical differentiation never mutates the
 * live simulation state. The first implementation uses central differences;
 * an mj_jacSite implementation can replace it without changing the solver.
 */
import * as THREE from 'three';
import { IKKinematics, IKPose } from './DampedLeastSquaresIK';
import { MujocoData, MujocoModel, MujocoModule } from './types';

export interface MuJoCoJointBinding {
    name: string;
    jointId: number;
    actuatorId: number;
    qposAddress: number;
    dofAddress: number;
    lowerLimit: number;
    upperLimit: number;
}

function quaternionFromSiteMatrix(matrix: ArrayLike<number>, offset: number): THREE.Quaternion {
    const transform = new THREE.Matrix4().set(
        matrix[offset], matrix[offset + 1], matrix[offset + 2], 0,
        matrix[offset + 3], matrix[offset + 4], matrix[offset + 5], 0,
        matrix[offset + 6], matrix[offset + 7], matrix[offset + 8], 0,
        0, 0, 0, 1,
    );
    return new THREE.Quaternion().setFromRotationMatrix(transform).normalize();
}

function rotationDifference(from: THREE.Quaternion, to: THREE.Quaternion): THREE.Vector3 {
    const q = to.clone().multiply(from.clone().invert()).normalize();
    if (q.w < 0) q.set(-q.x, -q.y, -q.z, -q.w);
    const sinHalf = Math.hypot(q.x, q.y, q.z);
    if (sinHalf < 1e-10) return new THREE.Vector3(2 * q.x, 2 * q.y, 2 * q.z);
    return new THREE.Vector3(q.x, q.y, q.z)
        .multiplyScalar(2 * Math.atan2(sinHalf, q.w) / sinHalf);
}

export class MuJoCoKinematics implements IKKinematics {
    readonly joints: MuJoCoJointBinding[];
    readonly lowerLimits: number[];
    readonly upperLimits: number[];
    private readonly scratch: MujocoData;
    private referenceQpos: Float64Array;
    private disposed = false;

    constructor(
        private readonly mujoco: MujocoModule,
        private readonly model: MujocoModel,
        private readonly liveData: MujocoData,
        readonly siteId: number,
        joints: MuJoCoJointBinding[],
        private readonly finiteDifferenceStep = 1e-5,
    ) {
        if (siteId < 0 || siteId >= model.nsite) throw new Error('Invalid MuJoCo TCP site id');
        if (joints.length === 0) throw new Error('IK requires at least one actuated joint');
        this.joints = joints;
        this.lowerLimits = joints.map(j => j.lowerLimit);
        this.upperLimits = joints.map(j => j.upperLimit);
        this.scratch = new mujoco.MjData(model);
        this.referenceQpos = new Float64Array(liveData.qpos);
    }

    /** Refresh non-arm coordinates and the current arm configuration. */
    captureReference(): number[] {
        this.referenceQpos = new Float64Array(this.liveData.qpos);
        return this.joints.map(j => this.liveData.qpos[j.qposAddress]);
    }

    private forward(joints: readonly number[]): IKPose {
        this.scratch.qpos.set(this.referenceQpos);
        for (let i = 0; i < this.joints.length; i++) {
            this.scratch.qpos[this.joints[i].qposAddress] = joints[i];
        }
        this.mujoco.mj_forward(this.model, this.scratch);
        const p = this.scratch.site_xpos;
        const m = this.scratch.site_xmat;
        return {
            position: new THREE.Vector3(p[this.siteId * 3], p[this.siteId * 3 + 1], p[this.siteId * 3 + 2]),
            quaternion: quaternionFromSiteMatrix(m, this.siteId * 9),
        };
    }

    evaluate(joints: readonly number[]) {
        const pose = this.forward(joints);
        const n = joints.length;
        const jacobian = Array.from({ length: 6 }, () => Array(n).fill(0));

        for (let col = 0; col < n; col++) {
            const plus = [...joints];
            const minus = [...joints];
            plus[col] += this.finiteDifferenceStep;
            minus[col] -= this.finiteDifferenceStep;
            const pPose = this.forward(plus);
            const mPose = this.forward(minus);

            const linear = pPose.position.sub(mPose.position)
                .multiplyScalar(0.5 / this.finiteDifferenceStep);
            const angular = rotationDifference(mPose.quaternion, pPose.quaternion)
                .multiplyScalar(0.5 / this.finiteDifferenceStep);

            jacobian[0][col] = linear.x;
            jacobian[1][col] = linear.y;
            jacobian[2][col] = linear.z;
            jacobian[3][col] = angular.x;
            jacobian[4][col] = angular.y;
            jacobian[5][col] = angular.z;
        }
        return { pose, jacobian };
    }

    /** Apply a solved arm configuration to controls, not directly to qpos. */
    applyControls(solution: readonly number[], data = this.liveData): void {
        for (let i = 0; i < Math.min(solution.length, this.joints.length); i++) {
            data.ctrl[this.joints[i].actuatorId] = solution[i];
        }
    }

    dispose(): void {
        if (this.disposed) return;
        this.disposed = true;
        this.scratch.delete();
    }
}

/**
 * Resolve a configured list of scalar hinge/slide joints to their MuJoCo
 * qpos, dof and actuator addresses. Names make the mapping independent of
 * model ordering and gripper actuators.
 */
export function resolveMuJoCoJoints(
    model: MujocoModel,
    getName: (address: number) => string,
    jointNames: readonly string[],
): MuJoCoJointBinding[] {
    const bindings: MuJoCoJointBinding[] = [];
    const usedActuators = new Set<number>();

    for (const name of jointNames) {
        let jointId = -1;
        for (let id = 0; id < model.njnt; id++) {
            if (getName(model.name_jntadr[id]) === name) {
                jointId = id;
                break;
            }
        }
        if (jointId < 0) throw new Error(`MuJoCo joint "${name}" not found`);

        // MuJoCo joint types: slide=2, hinge=3. Both have scalar qpos and dof.
        const type = model.jnt_type?.[jointId];
        if (type !== 2 && type !== 3) {
            throw new Error(`IK joint "${name}" must be a scalar hinge or slide joint`);
        }

        let actuatorId = -1;
        for (let id = 0; id < model.nu; id++) {
            const transmittedJoint = model.actuator_trnid[2 * id];
            if (transmittedJoint === jointId && !usedActuators.has(id)) {
                actuatorId = id;
                break;
            }
        }
        if (actuatorId < 0) throw new Error(`No direct joint actuator found for "${name}"`);
        usedActuators.add(actuatorId);

        const limited = model.jnt_limited?.[jointId] === 1;
        const range = model.jnt_range;
        bindings.push({
            name,
            jointId,
            actuatorId,
            qposAddress: model.jnt_qposadr[jointId],
            dofAddress: model.jnt_dofadr[jointId],
            lowerLimit: limited ? range[jointId * 2] : -Infinity,
            upperLimit: limited ? range[jointId * 2 + 1] : Infinity,
        });
    }
    return bindings;
}
