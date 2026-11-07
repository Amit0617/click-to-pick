# Robotics Simulator in Browser (Franka Panda)

A web-based robotics simulator running directly in the browser using MuJoCo physics compiled to WebAssembly (WASM), Three.js rendering, and multimodal spatial reasoning via Gemini and manual direct targeting ("Click to Pick").

## Architecture Evolution

```mermaid
flowchart LR
    subgraph root[" "]
        direction TB
        
        subgraph v0["Robotics simulator in browser v0"]
            direction TB
            A1["Loading MuJoCo Wasm as environment"]
            A2["Importing a robot (franka panda) in MuJoCo"]
            A3["Implementing Inverse Kinematics (analytical only for franka)"]
            A4["Integrating Gemini Spatial understanding<br/>(annotates objects in scene)"]
            
            A1 ~~~ A2 ~~~ A3 ~~~ A4
        end

        subgraph v1["Robotics simulator in browser v1"]
            direction TB
            B1["Loading MuJoCo Wasm as environment"]
            B2["Importing a robot (franka panda) in MuJoCo"]
            B3["Implementing Inverse Kinematics (analytical only for franka)"]
            
            subgraph bottomRow[" "]
                direction LR
                B4["Integrating Gemini<br/>Spatial understanding<br/>(annotates objects in scene)"]
                B5["Manually Clicking on object to Pick"]
            end
            
            B1 ~~~ B2 ~~~ B3 ~~~ bottomRow
        end

        subgraph v2["Robotics simulator in browser v2"]
            direction TB
            C1["Loading MuJoCo Wasm as environment"]
            C2["Importing a robot (franka panda) in MuJoCo"]
            
            subgraph aboveBottomRow[" "]
                direction LR
                C3["Implementing Inverse Kinematics (analytical only for franka)"]
                C4["PyRoKi Fast API backend with generalized IK"]
            end

            subgraph bottomRow1[" "]
                direction LR
                C5["Integrating Gemini<br/>Spatial understanding<br/>(annotates objects in scene)"]
                C6["Manually Clicking on object to Pick"]
            end
            
            C1 ~~~ C2 ~~~ aboveBottomRow ~~~ bottomRow1
        end

        subgraph v3["Robotics simulator in browser v3"]
            direction TB
            D1["Loading MuJoCo Wasm as environment"]
            
            subgraph belowTop[" "]
                direction LR
                D2["Importing a robot (franka panda) in MuJoCo"]
                D3["Importing UR5e with Robotiq 2f-85 gripper in MuJoCo"]
            end
            
            subgraph aboveBottomRow1[" "]
                direction LR
                D4["Implementing Inverse Kinematics (analytical only for franka)"]
                D5["PyRoKi Fast API backend with generalized IK"]
            end

            subgraph bottomRow2[" "]
                direction LR
                D6["Integrating Gemini<br/>Spatial understanding<br/>(annotates objects in scene)"]
                D7["Manually Clicking on object to Pick"]
            end
            
            D1 ~~~ belowTop ~~~ aboveBottomRow1 ~~~ bottomRow2
        end
   end

    %% Styles
    classDef default fill:#FFFFFF,stroke:#D9D9D9,stroke-width:1px,color:#222222,rx:6px;
    classDef greenCard fill:#E6F7ED,stroke:#34C759,stroke-width:1.5px,color:#1C1C1E,rx:6px;
    
    class B5 greenCard;
    class C4 greenCard;
    class D3 greenCard;
    
    style root fill:none,stroke:none;
    style v0 fill:#F9F9F9,stroke:#E5E5E5,stroke-width:1px,rx:10px;
    style v1 fill:#F9F9F9,stroke:#E5E5E5,stroke-width:1px,rx:10px;
    style v2 fill:#F9F9F9,stroke:#E5E5E5,stroke-width:1px,rx:10px;
    style v3 fill:#F9F9F9,stroke:#E5E5E5,stroke-width:1px,rx:10px;

```

`Robotics simulator in browser v0` is taken from [Xavier Plantaz from Google AI](https://dev.to/googleai/building-a-gemini-powered-robotics-simulator-in-the-browser-with-mujoco-wasm-hjj) 

## Getting Started

### Prerequisites
- Node.js (v18+)
- npm or pnpm

### Installation & Run Locally
1. Install dependencies:
```sh
npm install
# or pnpm install
```
2. Set the `GEMINI_API_KEY` in [.env.local](.env.local) to your Gemini API key

3. Run Backend in another terminal (leave that running)
```sh
cd server
uv sync
uv run python pyroki_server.py
```

4. Run the app (in another terminal tab):
   `npm run dev`/ `pnpm run dev` at the root of project.

## Local MuJoCo DLS IK (experimental)

The `dls` solver runs in the browser and uses the loaded MuJoCo model as its
kinematic source of truth. It resolves the configured arm joints by name,
reads their `qpos`/actuator addresses from the compiled model, and evaluates
the TCP pose on a separate scratch `MjData`. The live simulation state is not
modified during optimization; only the resulting actuator controls are
written after a converged solution is found.

The first implementation uses a central finite-difference geometric Jacobian
and an adaptive damped-least-squares update. This is intentionally a portable
baseline while the `mj_jacSite` WASM export is verified. It is suitable for
interactive prototyping, but finite differences require multiple forward
passes per iteration and may be slower than a native MuJoCo Jacobian.

Select **DLS** in the IK solver panel to compare it with PyRoKi. PyRoKi remains
the default/reference solver. The local solver currently handles scalar hinge
and slide arm joints, joint limits, and 6D end-effector pose error. It does not
perform collision avoidance or trajectory planning.
