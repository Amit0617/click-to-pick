/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import * as THREE from 'three';
import { TransformControls } from 'three/examples/jsm/controls/TransformControls.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { calculateAnalyticalIK } from './FrankaAnalyticalIK';
import { ROBOT_CONFIGS, RobotSpec } from './robots';
import { MujocoData, MujocoModel, MujocoModule } from './types';

function squaredDistance(arr1: number[], arr2: number[]) {
    let sum = 0;
    for (let i = 0; i < Math.min(arr1.length, arr2.length); i++) {
        sum += (arr1[i] - arr2[i]) ** 2;
    }
    return sum;
}

export type IkSolverType = 'analytical' | 'pyroki';

export interface IkSolveStats {
    method: 'analytical' | 'pyroki';
    timeMs: number;
    robotId?: string;
}

/**
 * IkSystem
 * Handles Inverse Kinematics calculations using either the Franka 7-DOF analytical solver
 * or the PyRoKi (JAX-based differentiable optimization) FastAPI backend for arbitrary robots (Franka, UR5e).
 */
export class IkSystem {
    target: THREE.Group;
    control: TransformControls;
    
    calculating = false;
    gripperSiteId = -1;

    currentRobotId = 'franka_panda';
    currentSpec: RobotSpec = ROBOT_CONFIGS.franka_panda;
    dof = 7;

    solverType: IkSolverType = 'pyroki';
    lastSolveStats: IkSolveStats | null = null;
    onSolveCallback: ((stats: IkSolveStats) => void) | null = null;

    private isAsyncSolving = false;
    private lastRequestedTarget: { position: THREE.Vector3; quaternion: THREE.Quaternion } | null = null;
    private qNeutral = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]; // Preferred "home" pose
    
    // Joint 7 parameters for redundancy resolution in Franka analytical solver
    private readonly q7Min = -2.8973;
    private readonly q7Max = 2.8973;
    private readonly q7Step = 0.1; 

    constructor(mujoco: MujocoModule, camera: THREE.Camera, domElement: HTMLElement, orbitControls: OrbitControls) {
        this.target = new THREE.Group();
        this.target.name = "IK Target";
        
        // Visual aid for target
        const axes = new THREE.AxesHelper(0.2);
        this.target.add(axes);
        
        this.control = new TransformControls(camera, domElement);
        this.control.addEventListener('dragging-changed', (event) => {
            const e = event as unknown as { value: boolean };
            orbitControls.enabled = !e.value;
        });
        this.control.attach(this.target);
    }

    setRobot(robotId: string) {
        this.currentRobotId = robotId;
        this.currentSpec = ROBOT_CONFIGS[robotId] || ROBOT_CONFIGS.franka_panda;
        this.dof = this.currentSpec.dof;
        this.qNeutral = [...this.currentSpec.homeJoints];

        if (!this.currentSpec.availableSolvers.includes(this.solverType)) {
            this.solverType = this.currentSpec.defaultSolver;
        }
    }

    setSolverType(type: IkSolverType) {
        if (this.currentSpec.availableSolvers.includes(type)) {
            this.solverType = type;
        }
    }
    
    init(mjModel: MujocoModel, isDouble: boolean) {
        // Reset internal state if needed
    }
    
    syncToSite(mjData: MujocoData) {
        if (this.gripperSiteId === -1) return;
        // Get site position and rotation from MuJoCo
        const sitePos = mjData.site_xpos.subarray(this.gripperSiteId * 3, this.gripperSiteId * 3 + 3);
        const siteMat = mjData.site_xmat.subarray(this.gripperSiteId * 9, this.gripperSiteId * 9 + 9);
        
        this.target.position.set(sitePos[0], sitePos[1], sitePos[2]);
        
        const m = new THREE.Matrix4().set(
            siteMat[0], siteMat[1], siteMat[2], 0,
            siteMat[3], siteMat[4], siteMat[5], 0,
            siteMat[6], siteMat[7], siteMat[8], 0,
            0, 0, 0, 1
        );
        this.target.quaternion.setFromRotationMatrix(m);
    }

    /**
     * Solves IK using analytical geometric formula (Franka Panda only).
     */
    solveAnalytical(pos: THREE.Vector3, quat: THREE.Quaternion, currentQ: number[]): number[] | null {
        if (this.currentRobotId !== 'franka_panda') {
            console.warn(`[Analytical IK] Analytical closed-form solver is only available for Franka Panda 7-DOF. Using PyRoKi for ${this.currentSpec.name}.`);
            return null;
        }

        const t0 = performance.now();
        this.target.position.copy(pos);
        this.target.quaternion.copy(quat);
        this.target.updateMatrixWorld();
        const transform = this.target.matrixWorld;

        const euler = new THREE.Euler().setFromQuaternion(quat);
        const eulerDeg = [
            (euler.x * 180 / Math.PI).toFixed(1),
            (euler.y * 180 / Math.PI).toFixed(1),
            (euler.z * 180 / Math.PI).toFixed(1)
        ];

        console.log(`%c[Analytical IK] Solving target`, 'color: #3b82f6; font-weight: bold;', {
            position: [Number(pos.x.toFixed(4)), Number(pos.y.toFixed(4)), Number(pos.z.toFixed(4))],
            quaternion: [Number(quat.x.toFixed(4)), Number(quat.y.toFixed(4)), Number(quat.z.toFixed(4)), Number(quat.w.toFixed(4))],
            eulerDeg: `[${eulerDeg.join(', ')}]°`,
            currentJoints: currentQ.map(q => Number(q.toFixed(3)))
        });

        // --- Redundancy Resolution Strategy ---
        // 1. Try solution with current q7 (fastest, keeps continuity)
        // 2. If valid, refine locally.
        // 3. If not, scan full range.

        // Weights for cost function: 
        // Minimize (Distance to Current Joints) + (Distance to Neutral Joints)        
        const alpha = 1.0; // Continuity weight
        const beta = 0.05;  // Neutrality weight

        let bestSolution: number[] | null = null;
        let minCost = Infinity;
        let evaluatedCount = 0;

        const processCandidateQ7 = (q7: number) => {
             const solutions = calculateAnalyticalIK(transform, q7);
             for (const sol of solutions) {
                 evaluatedCount++;
                 const distCurrent = squaredDistance(sol, currentQ);
                 const distNeutral = squaredDistance(sol, this.qNeutral);
                 const cost = alpha * distCurrent + beta * distNeutral;
                 
                 if (cost < minCost) {
                     minCost = cost;
                     bestSolution = sol;
                 }
             }
        };

        // 1. Try current q7
        const currentQ7 = currentQ[6];
        processCandidateQ7(currentQ7);

        // 2. If no solution or to optimize, scan nearby
        // Simple optimization: Scan range around currentQ7
        const searchRange = 0.5;
        for (let q7 = Math.max(this.q7Min, currentQ7 - searchRange); q7 <= Math.min(this.q7Max, currentQ7 + searchRange); q7 += this.q7Step) {
            processCandidateQ7(q7);
        }

        // 3. Fallback: If still no solution, scan entire range (global search)
        if (!bestSolution) {
             for (let q7 = this.q7Min; q7 <= this.q7Max; q7 += this.q7Step * 2) {
                 processCandidateQ7(q7);
             }
        }

        const elapsed = performance.now() - t0;
        this.lastSolveStats = { method: 'analytical', timeMs: elapsed, robotId: this.currentRobotId };
        if (this.onSolveCallback) {
            this.onSolveCallback(this.lastSolveStats);
        }

        if (bestSolution) {
            const degs = (bestSolution as number[]).map(r => (r * 180 / Math.PI).toFixed(1) + '°');
            console.log(`%c[Analytical IK] SUCCESS (${elapsed.toFixed(2)}ms, ${evaluatedCount} candidates evaluated)`, 'color: #10b981; font-weight: bold;', {
                jointsRad: (bestSolution as number[]).map(r => Number(r.toFixed(4))),
                jointsDeg: degs
            });
        } else {
            console.warn(`%c[Analytical IK] FAILED (${elapsed.toFixed(2)}ms): Target out of physical reach or violates joint limits.`, 'color: #ef4444; font-weight: bold;', {
                targetPosition: [Number(pos.x.toFixed(4)), Number(pos.y.toFixed(4)), Number(pos.z.toFixed(4))],
                distFromBase: Math.sqrt(pos.x*pos.x + pos.y*pos.y + pos.z*pos.z).toFixed(3) + 'm (Max reach ~0.855m)'
            });
        }

        return bestSolution;
    }

    /**
     * Solves IK using PyRoKi FastAPI backend with generalized robot morphology support.
     */
    async solvePyroki(pos: THREE.Vector3, quat: THREE.Quaternion, currentQ: number[]): Promise<number[] | null> {
        const t0 = performance.now();
        console.log(`%c[PyRoKi IK] Dispatching request to JAX solver`, 'color: #8b5cf6; font-weight: bold;', {
            position: [Number(pos.x.toFixed(4)), Number(pos.y.toFixed(4)), Number(pos.z.toFixed(4))],
            quaternion: [Number(quat.x.toFixed(4)), Number(quat.y.toFixed(4)), Number(quat.z.toFixed(4)), Number(quat.w.toFixed(4))]
        });

        try {
            const res = await fetch('/api/ik/solve', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    robot_id: this.currentSpec.pyrokiRobotId,
                    position: [pos.x, pos.y, pos.z],
                    quaternion: [quat.x, quat.y, quat.z, quat.w],
                    current_joints: currentQ.slice(0, this.dof)
                })
            });

            if (!res.ok) {
                if (this.currentSpec.availableSolvers.includes('analytical')) {
                    console.warn(`%c[PyRoKi IK] HTTP ${res.status} error, falling back to analytical solver`, 'color: #f59e0b; font-weight: bold;');
                    return this.solveAnalytical(pos, quat, currentQ);
                }
                return null;
            }

            const data = await res.json();
            const totalElapsed = performance.now() - t0;

            if (data.success && Array.isArray(data.joints)) {
                const elapsed = data.computation_time_ms || totalElapsed;
                this.lastSolveStats = { method: 'pyroki', timeMs: elapsed, robotId: this.currentRobotId };
                if (this.onSolveCallback) {
                    this.onSolveCallback(this.lastSolveStats);
                }
                const posErrStr = data.pos_error_m != null ? `${(data.pos_error_m * 1000).toFixed(2)}mm` : 'verified';
                const rotErrStr = data.rot_error_rad != null ? `${data.rot_error_rad.toFixed(4)}rad` : 'verified';
                console.log(`%c[PyRoKi IK] SUCCESS (JAX: ${elapsed}ms, roundtrip: ${totalElapsed.toFixed(1)}ms | pos_err: ${posErrStr}, rot_err: ${rotErrStr})`, 'color: #10b981; font-weight: bold;', {
                    jointsRad: data.joints.map((r: number) => Number(r.toFixed(4))),
                    posErrorM: data.pos_error_m,
                    rotErrorRad: data.rot_error_rad
                });
                return data.joints;
            } else {
                if (this.currentSpec.availableSolvers.includes('analytical')) {
                    console.warn(`%c[PyRoKi IK] UNSUCCESSFUL: ${data.error || 'Tolerance limit exceeded'} (pos_err: ${(data.pos_error_m * 1000)?.toFixed(1)}mm). Falling back to Analytical solver...`, 'color: #f59e0b; font-weight: bold;', data);
                    return this.solveAnalytical(pos, quat, currentQ);
                }
                return null;
            }
        } catch (err) {
            console.warn(`%c[PyRoKi IK] Request failed (${err}), checking available fallbacks...`, 'color: #f59e0b; font-weight: bold;');
            if (this.currentSpec.availableSolvers.includes('analytical')) {
                return this.solveAnalytical(pos, quat, currentQ);
            }
            return null;
        }
    }

    /**
     * Unified asynchronous solve method respecting the currently selected solverType.
     */
    async solveAsync(pos: THREE.Vector3, quat: THREE.Quaternion, currentQ: number[]): Promise<number[] | null> {
        if (this.solverType === 'pyroki') {
            return await this.solvePyroki(pos, quat, currentQ);
        }
        return this.solveAnalytical(pos, quat, currentQ);
    }

    /**
     * Synchronous solve method for drop-in compatibility with existing pipeline callers.
     * Uses analytical solution directly.
     */
    solve(pos: THREE.Vector3, quat: THREE.Quaternion, currentQ: number[]): number[] | null {
        return this.solveAnalytical(pos, quat, currentQ);
    }
    
    update(mjModel: MujocoModel, mjData: MujocoData) {
        if (!this.calculating) return;
        
        // Prepare current state for arm joints
        const currentQ: number[] = [];
        for (let i = 0; i < this.dof; i++) currentQ.push(mjData.qpos[i]);

        if (this.solverType === 'pyroki') {
            const targetChanged = !this.lastRequestedTarget
                || this.target.position.distanceToSquared(this.lastRequestedTarget.position) > 1e-6
                || this.target.quaternion.angleTo(this.lastRequestedTarget.quaternion) > 0.01;

            if (!this.isAsyncSolving && targetChanged) {
                this.isAsyncSolving = true;
                this.lastRequestedTarget = {
                    position: this.target.position.clone(),
                    quaternion: this.target.quaternion.clone()
                };
                this.solvePyroki(this.target.position, this.target.quaternion, currentQ)
                    .then((solution) => {
                        if (solution) {
                            for (let i = 0; i < Math.min(solution.length, this.dof); i++) {
                                mjData.ctrl[i] = solution[i];
                            }
                        }
                    })
                    .finally(() => {
                        this.isAsyncSolving = false;
                    });
            }
        } else {
            // Solve analytical
            const solution = this.solveAnalytical(this.target.position, this.target.quaternion, currentQ);
            if (solution) {
                for (let i = 0; i < Math.min(solution.length, this.dof); i++) {
                    mjData.ctrl[i] = solution[i];
                }
            }
        }
    }
    
    setCalculating(enabled: boolean) {
        this.calculating = enabled;
    }
    
    setGizmoVisible(visible: boolean) {
        this.control.getHelper().visible = visible;
        this.control.enabled = visible;
    }
    
    setTargetVisible(visible: boolean) {
        this.target.visible = visible;
    }
    
    setMode(mode: string) {
        // Only freeform implemented
    }
    
    isActuatorIkControlled(id: number) {
        return id >= 0 && id < this.dof;
    }
    
    dispose() {
        this.control.dispose();
    }
}
