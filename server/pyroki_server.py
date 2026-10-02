import time
import os
import sys
from typing import List, Optional, Dict, Any
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import jax
import jax.numpy as jnp
import jaxlie
import jaxls
import pyroki as pk
from robot_descriptions.loaders.yourdfpy import load_robot_description

app = FastAPI(title="PyRoKi Generalized IK Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RobotModelContext:
    def __init__(self, robot_id: str, description_name: str, target_link_candidates: List[str], q_home: List[float], arm_dof: int, tcp_offset: float = 0.0):
        self.robot_id = robot_id
        self.description_name = description_name
        self.arm_dof = arm_dof
        self.tcp_offset = tcp_offset
        self.q_home = jnp.array(q_home, dtype=jnp.float32)
        
        print(f"[{robot_id}] Loading robot description '{description_name}'...")
        self.urdf = load_robot_description(description_name)
        self.robot = pk.Robot.from_urdf(self.urdf)
        
        # Find matching target link
        target_name = None
        for candidate in target_link_candidates:
            if candidate in self.robot.links.names:
                target_name = candidate
                break
        if target_name is None:
            # Fallback to last link
            target_name = self.robot.links.names[-1]
            
        self.target_link_name = target_name
        self.target_link_index = jnp.array(self.robot.links.names.index(target_name))
        print(f"[{robot_id}] Loaded successfully! Target link '{target_name}' at index {int(self.target_link_index)} (Actuated joints: {self.robot.joints.num_actuated_joints})")
        
        # Compile solver function for this specific robot topology
        self._build_solver()

    def _build_solver(self):
        robot = self.robot
        target_link_index = self.target_link_index

        @jax.jit
        def _solve(target_pos: jax.Array, target_wxyz: jax.Array, rest_q: jax.Array, init_q: jax.Array):
            joint_var = robot.joint_var_cls(0)
            costs = [
                pk.costs.pose_cost_analytic_jac(
                    robot,
                    joint_var,
                    jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(target_wxyz), target_pos),
                    target_link_index,
                    pos_weight=60.0,
                    ori_weight=15.0,
                ),
                pk.costs.limit_constraint(robot, joint_var),
                pk.costs.rest_cost(joint_var, rest_pose=rest_q, weight=0.2),
            ]
            sol = (
                jaxls.LeastSquaresProblem(costs=costs, variables=[joint_var])
                .analyze()
                .solve(
                    initial_vals=jaxls.VarValues.make([joint_var.with_value(init_q)]),
                    verbose=False,
                    linear_solver="dense_cholesky",
                    trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
                )
            )
            q_sol = sol[joint_var]
            fk = robot.forward_kinematics(q_sol)
            se3 = jaxlie.SE3(fk[target_link_index])
            pos_err = jnp.linalg.norm(se3.translation() - target_pos)
            target_so3 = jaxlie.SO3(target_wxyz)
            rot_err = jnp.linalg.norm((target_so3.inverse() @ se3.rotation()).log())
            return q_sol, pos_err, rot_err

        self._solve_fn = _solve

    def warmup(self):
        print(f"[{self.robot_id}] Warming up JAX JIT compilation...")
        t0 = time.time()
        _pos = jnp.array([0.4, 0.0, 0.2], dtype=jnp.float32)
        _wxyz = jnp.array([0.0, 1.0, 0.0, 0.0], dtype=jnp.float32)
        _sol, _perr, _rerr = self._solve_fn(_pos, _wxyz, self.q_home, self.q_home)
        _sol.block_until_ready()
        print(f"[{self.robot_id}] JIT Warmup complete in {time.time() - t0:.2f}s! Initial pos err: {float(_perr)*1000:.2f}mm")

# Preload supported manipulators
ROBOT_REGISTRY: Dict[str, RobotModelContext] = {}

# 1. Franka Panda (7 arm joints + 1 gripper joint)
try:
    panda_ctx = RobotModelContext(
        robot_id="franka_panda",
        description_name="panda_description",
        target_link_candidates=["panda_hand_tcp", "panda_hand", "panda_link8"],
        q_home=[1.707, -1.754, 0.003, -2.702, 0.003, 0.951, 2.490, 0.000],
        arm_dof=7
    )
    panda_ctx.warmup()
    ROBOT_REGISTRY["franka_panda"] = panda_ctx
except Exception as e:
    print(f"Error loading franka_panda: {e}")

# 2. Universal Robots UR5e (6 revolute arm joints) with Robotiq 2F-85 gripper
try:
    ur5e_ctx = RobotModelContext(
        robot_id="ur5e",
        description_name="ur5e_description",
        target_link_candidates=["tool0", "wrist_3_link", "flange"],
        q_home=[1.5, -2.2708, 2.2708, -1.5708, -1.5708, 0.0],
        arm_dof=6,
        tcp_offset=0.135
    )
    ur5e_ctx.warmup()
    ROBOT_REGISTRY["ur5e"] = ur5e_ctx
except Exception as e:
    print(f"Error loading ur5e: {e}")

class IKRequest(BaseModel):
    robot_id: Optional[str] = "franka_panda"
    position: List[float] # [x, y, z]
    quaternion: List[float] # Three.js [x, y, z, w]
    current_joints: List[float] # arm joints (7 for Franka, 6 for UR5e)

class IKResponse(BaseModel):
    success: bool
    robot_id: str = "franka_panda"
    joints: Optional[List[float]] = None
    computation_time_ms: float
    pos_error_m: Optional[float] = None
    rot_error_rad: Optional[float] = None
    method: str = "pyroki"
    error: Optional[str] = None

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "solver": "pyroki",
        "available_robots": list(ROBOT_REGISTRY.keys()),
        "robot_details": {
            rid: {
                "dof": ctx.arm_dof,
                "target_link": ctx.target_link_name,
                "actuated_joints": ctx.robot.joints.num_actuated_joints
            }
            for rid, ctx in ROBOT_REGISTRY.items()
        }
    }

@app.post("/api/ik/solve", response_model=IKResponse)
def solve_ik(req: IKRequest):
    t_start = time.time()
    robot_id = req.robot_id or "franka_panda"
    
    if robot_id not in ROBOT_REGISTRY:
        # Fallback to first available or error
        if "franka_panda" in ROBOT_REGISTRY:
            robot_id = "franka_panda"
        elif len(ROBOT_REGISTRY) > 0:
            robot_id = list(ROBOT_REGISTRY.keys())[0]
        else:
            return IKResponse(
                success=False,
                robot_id=robot_id,
                computation_time_ms=0.0,
                error=f"Robot model '{robot_id}' is not loaded."
            )

    ctx = ROBOT_REGISTRY[robot_id]

    try:
        # Convert position
        pos = jnp.array(req.position, dtype=jnp.float32)
        
        # Convert quaternion from Three.js (x, y, z, w) to jaxlie (w, x, y, z)
        qx, qy, qz, qw = req.quaternion
        norm = float(np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw))
        if norm > 1e-6:
            qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm
        wxyz = jnp.array([qw, qx, qy, qz], dtype=jnp.float32)
        
        # Account for tool / gripper TCP offset if target_link is at the mounting flange
        if ctx.tcp_offset > 0.0:
            target_so3 = jaxlie.SO3(wxyz)
            tool_dir = target_so3.apply(jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32))
            pos_link = pos - ctx.tcp_offset * tool_dir
        else:
            pos_link = pos

        # Format joints to match total actuated joints required by the model
        q_curr = list(req.current_joints)
        total_actuated = ctx.robot.joints.num_actuated_joints
        
        # Pad or slice to match robot expected joint count
        if len(q_curr) < total_actuated:
            q_curr.extend([0.0] * (total_actuated - len(q_curr)))
        elif len(q_curr) > total_actuated:
            q_curr = q_curr[:total_actuated]
            
        current_q = jnp.array(q_curr, dtype=jnp.float32)
        
        # 1st attempt: solve from current joint posture for trajectory continuity
        sol, pos_err_jnp, rot_err_jnp = ctx._solve_fn(pos_link, wxyz, current_q, current_q)
        sol.block_until_ready()
        pos_err = float(pos_err_jnp)
        rot_err = float(rot_err_jnp)
        
        # Physical tolerances for robotic manipulation
        POS_TOLERANCE_M = 0.030 # 30 mm
        ROT_TOLERANCE_RAD = 0.40

        # 2nd attempt: if local minimum or joint limits trapped optimizer, restart from home pose
        if pos_err > POS_TOLERANCE_M:
            sol_home, pos_err_home_jnp, rot_err_home_jnp = ctx._solve_fn(pos_link, wxyz, current_q, ctx.q_home)
            sol_home.block_until_ready()
            pos_err_home = float(pos_err_home_jnp)
            rot_err_home = float(rot_err_home_jnp)
            if pos_err_home < pos_err:
                sol, pos_err, rot_err = sol_home, pos_err_home, rot_err_home

        elapsed_ms = (time.time() - t_start) * 1000.0
        is_success = (pos_err <= POS_TOLERANCE_M) and (rot_err <= ROT_TOLERANCE_RAD)
        
        if is_success:
            solved_joints = [float(x) for x in sol[:ctx.arm_dof]]
            return IKResponse(
                success=True,
                robot_id=robot_id,
                joints=solved_joints,
                computation_time_ms=round(elapsed_ms, 2),
                pos_error_m=round(pos_err, 4),
                rot_error_rad=round(rot_err, 4),
                method="pyroki"
            )
        else:
            return IKResponse(
                success=False,
                robot_id=robot_id,
                joints=None,
                computation_time_ms=round(elapsed_ms, 2),
                pos_error_m=round(pos_err, 4),
                rot_error_rad=round(rot_err, 4),
                method="pyroki",
                error=f"Tolerance exceeded: pos error {pos_err * 1000:.1f} mm, rot error {rot_err:.3f} rad"
            )
    except Exception as e:
        elapsed_ms = (time.time() - t_start) * 1000.0
        return IKResponse(
            success=False,
            robot_id=robot_id,
            joints=None,
            computation_time_ms=round(elapsed_ms, 2),
            method="pyroki",
            error=str(e)
        )

if __name__ == "__main__":
    port = int(os.environ.get("PYROKI_PORT", 5050))
    print(f"Starting PyRoKi Generalized FastAPI server on 127.0.0.1:{port}...")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
