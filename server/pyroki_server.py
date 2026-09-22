import time
import os
import sys
from typing import List, Optional
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

app = FastAPI(title="PyRoKi IK Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Franka Panda robot model
print("Loading Franka Panda description...")
urdf = load_robot_description("panda_description")
robot = pk.Robot.from_urdf(urdf)
target_link_index = jnp.array(robot.links.names.index("panda_hand_tcp"))
print(f"Robot loaded! Target link 'panda_hand_tcp' index: {int(target_link_index)}")

# Franka Panda default home/neutral joint configuration
Q_HOME = jnp.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.0], dtype=jnp.float32)

@jax.jit
def _solve_and_verify_ik(target_pos: jax.Array, target_wxyz: jax.Array, rest_q: jax.Array, init_q: jax.Array):
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

# Warm-up JIT compilation on server startup
print("Warming up JAX JIT compilation...")
_p0 = jnp.array([0.4, 0.0, 0.2], dtype=jnp.float32)
_w0 = jnp.array([0.0, 1.0, 0.0, 0.0], dtype=jnp.float32)
_q0 = jnp.array([1.707, -1.754, 0.003, -2.702, 0.003, 0.951, 2.490, 0.0], dtype=jnp.float32)
_w_start = time.time()
_sol, _perr, _rerr = _solve_and_verify_ik(_p0, _w0, _q0, _q0)
_sol.block_until_ready()
print(f"JIT Warm-up complete in {time.time() - _w_start:.2f}s! Initial error: {_perr*1000:.2f}mm")

class IKRequest(BaseModel):
    position: List[float] # [x, y, z]
    quaternion: List[float] # Three.js [x, y, z, w]
    current_joints: List[float] # 7 joints

class IKResponse(BaseModel):
    success: bool
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
        "robot": "franka_panda",
        "actuated_joints": robot.joints.num_actuated_joints
    }

@app.post("/api/ik/solve", response_model=IKResponse)
def solve_ik(req: IKRequest):
    t_start = time.time()
    try:
        # Convert position
        pos = jnp.array(req.position, dtype=jnp.float32)
        
        # Convert quaternion from Three.js (x, y, z, w) to jaxlie (w, x, y, z)
        qx, qy, qz, qw = req.quaternion
        norm = float(np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw))
        if norm > 1e-6:
            qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm
        wxyz = jnp.array([qw, qx, qy, qz], dtype=jnp.float32)
        
        # Prepare 8 joints (7 arm joints + 1 gripper joint)
        q_curr = list(req.current_joints)
        if len(q_curr) == 7:
            q_curr.append(0.0)
        current_q = jnp.array(q_curr, dtype=jnp.float32)
        
        # 1st attempt: solve from current joint posture for trajectory continuity
        sol, pos_err_jnp, rot_err_jnp = _solve_and_verify_ik(pos, wxyz, current_q, current_q)
        sol.block_until_ready()
        pos_err = float(pos_err_jnp)
        rot_err = float(rot_err_jnp)
        
        # Maximum allowed tolerance for robotic grasping: 25 mm
        POS_TOLERANCE_M = 0.025
        ROT_TOLERANCE_RAD = 0.35

        # 2nd attempt: if local minimum or joint limits trapped the optimizer, restart from neutral home
        if pos_err > POS_TOLERANCE_M:
            sol_home, pos_err_home_jnp, rot_err_home_jnp = _solve_and_verify_ik(pos, wxyz, current_q, Q_HOME)
            sol_home.block_until_ready()
            pos_err_home = float(pos_err_home_jnp)
            rot_err_home = float(rot_err_home_jnp)
            if pos_err_home < pos_err:
                sol, pos_err, rot_err = sol_home, pos_err_home, rot_err_home

        elapsed_ms = (time.time() - t_start) * 1000.0
        
        # Verify convergence against physical tolerance
        is_success = (pos_err <= POS_TOLERANCE_M) and (rot_err <= ROT_TOLERANCE_RAD)
        
        if is_success:
            solved_joints = [float(x) for x in sol[:7]]
            return IKResponse(
                success=True,
                joints=solved_joints,
                computation_time_ms=round(elapsed_ms, 2),
                pos_error_m=round(pos_err, 4),
                rot_error_rad=round(rot_err, 4),
                method="pyroki"
            )
        else:
            return IKResponse(
                success=False,
                joints=None,
                computation_time_ms=round(elapsed_ms, 2),
                pos_error_m=round(pos_err, 4),
                rot_error_rad=round(rot_err, 4),
                method="pyroki",
                error=f"Tolerance exceeded: position error is {pos_err * 1000:.1f} mm (limit {POS_TOLERANCE_M * 1000:.0f} mm), rotation error is {rot_err:.3f} rad"
            )
    except Exception as e:
        elapsed_ms = (time.time() - t_start) * 1000.0
        return IKResponse(
            success=False,
            joints=None,
            computation_time_ms=round(elapsed_ms, 2),
            method="pyroki",
            error=str(e)
        )

if __name__ == "__main__":
    port = int(os.environ.get("PYROKI_PORT", 5050))
    print(f"Starting PyRoKi FastAPI server on 127.0.0.1:{port}...")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
