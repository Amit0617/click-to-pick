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

@jax.jit
def _solve_ik_jax(target_pos: jax.Array, target_wxyz: jax.Array, current_q: jax.Array) -> jax.Array:
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
        pk.costs.rest_cost(joint_var, rest_pose=current_q, weight=0.5),
    ]
    sol = (
        jaxls.LeastSquaresProblem(costs=costs, variables=[joint_var])
        .analyze()
        .solve(
            initial_vals=jaxls.VarValues.make([joint_var.with_value(current_q)]),
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
        )
    )
    return sol[joint_var]

# Warm-up JIT compilation on server startup
print("Warming up JAX JIT compilation...")
_p0 = jnp.array([0.4, 0.0, 0.2])
_w0 = jnp.array([0.0, 1.0, 0.0, 0.0])
_q0 = jnp.array([1.707, -1.754, 0.003, -2.702, 0.003, 0.951, 2.490, 0.0])
_w_start = time.time()
_res = _solve_ik_jax(_p0, _w0, _q0).block_until_ready()
print(f"JIT Warm-up complete in {time.time() - _w_start:.2f}s!")

class IKRequest(BaseModel):
    position: List[float] # [x, y, z]
    quaternion: List[float] # Three.js [x, y, z, w]
    current_joints: List[float] # 7 joints

class IKResponse(BaseModel):
    success: bool
    joints: Optional[List[float]] = None
    computation_time_ms: float
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
        
        # Solve
        sol = _solve_ik_jax(pos, wxyz, current_q).block_until_ready()
        solved_joints = [float(x) for x in sol[:7]]
        
        elapsed_ms = (time.time() - t_start) * 1000.0
        return IKResponse(
            success=True,
            joints=solved_joints,
            computation_time_ms=round(elapsed_ms, 2),
            method="pyroki"
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
