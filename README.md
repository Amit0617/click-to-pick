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
            A2["Importing a robot in MuJoCo"]
            A3["Implementing Inverse Kinematics (only for franka)"]
            A4["Integrating Gemini Spatial understanding<br/>(annotates objects in scene)"]
            
            A1 ~~~ A2 ~~~ A3 ~~~ A4
        end

        subgraph v1["Robotics simulator in browser v1"]
            direction TB
            B1["Loading MuJoCo Wasm as environment"]
            B2["Importing a robot in MuJoCo"]
            B3["Implementing Inverse Kinematics (only for franka)"]
            
            subgraph bottomRow[" "]
                direction LR
                B4["Integrating Gemini<br/>Spatial understanding<br/>(annotates objects in scene)"]
                B5["Manually Clicking on object to Pick"]
            end
            
            B1 ~~~ B2 ~~~ B3 ~~~ bottomRow
        end
   end

    %% Styles
    classDef default fill:#FFFFFF,stroke:#D9D9D9,stroke-width:1px,color:#222222,rx:6px;
    classDef greenCard fill:#E6F7ED,stroke:#34C759,stroke-width:1.5px,color:#1C1C1E,rx:6px;
    
    class B5 greenCard;
    
    style root fill:none,stroke:none;
    style v0 fill:#F9F9F9,stroke:#E5E5E5,stroke-width:1px,rx:10px;
    style v1 fill:#F9F9F9,stroke:#E5E5E5,stroke-width:1px,rx:10px;
    style bottomRow fill:none,stroke:none;
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
source .venv/bin/activate
python3 pyroki_server.py
```

4. Run the app (in another terminal tab):
   `npm run dev`/ `pnpm run dev` at the root of project.

### Run and deploy your AI Studio app

This contains everything you need to run your app locally.

View your app in AI Studio: https://ai.studio/apps/2dfef720-ed29-44b6-a6dc-7940694b09f5
