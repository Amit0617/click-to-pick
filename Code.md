# Franka Panda | Robotics Simulator & Pick and Place - Codebase Documentation

This project demonstrates both an **Embodied Reasoning AI loop** (via Google Gemini Vision models) and a direct **Click to Pick manual visual targeting mode** to guide a Franka Emika Panda 7-DOF robotic arm in picking up objects. The application combines React for the UI, Three.js for rendering, and MuJoCo (via WebAssembly) for physics simulation.

---

## Systems Overview

1. **Frontend & Control Interface (React)**:
   - Manages simulation states, user interactions, targeting mode selection, camera transitions, and visual interaction logs.
   - Provides two targeting modes:
     - **Click to Pick (Manual Visual)**: Captures a clean top-down orthographic snapshot of the table where users click exact points or draw bounding boxes with SVG targeting reticles.
     - **Gemini Embodied Reasoning (AI)**: Sends prompts and snapshots to Gemini models (`gemini-robotics-er-2-preview`, `gemini-2.5-flash`) for automated semantic detection.
   - Displays **Picked Items History** with statuses (`Targeted` vs `Picked`), timestamps, and modal detail views.

2. **Visualization & Scene Projection (Three.js)**:
   - Renders the Franka Panda robot arm, colored blocks, table, trays, and floor reflections.
   - Synchronizes every frame with MuJoCo physics state (`mjData`).
   - Translates 2D canvas pixel coordinates $(x, y)$ into 3D world space coordinates via raycasting (`project2DTo3D`).

3. **Physics Simulation (MuJoCo WebAssembly)**:
   - Runs `mujoco-js` compiled to WebAssembly directly inside the browser.
   - Loads the Franka Emika Panda MJCF model and assets from the DeepMind Menagerie.
   - Performs rigid-body collision detection, contact dynamics, gravity, and actuator dynamics.

4. **Kinematics & Motion Control**:
   - **Inverse Kinematics (`FrankaAnalyticalIK.ts`)**: Closed-form analytical inverse kinematics solver specialized for the Franka Emika Panda 7-DOF arm.
   - **IK Management (`IkSystem.ts`)**: Applies joint limits, resolves redundancy, and feeds computed joint angles into the simulation.
   - **Trajectory Execution (`SequenceAnimator.ts`)**: Orchestrates the multi-stage pick-and-place sequence:
     1. Move to pre-grasp hover position above target.
     2. Open two-finger parallel gripper.
     3. Lower to grasp position.
     4. Close gripper to grasp target.
     5. Lift object vertically.
     6. Move to sorting tray drop zone.
     7. Open gripper to release object.
     8. Return to ready home position.

---

## File Structure & Responsibilities

### Core Application
- **`index.tsx`**: Application bootstrap and React DOM root mounting.
- **`App.tsx`**: Main controller component.
  - Initializes `MujocoSim`.
  - Manages targeting modes, logs, and camera states.
  - Handles the dual targeting pipeline:
    - AI flow: `handleErSend`
    - Visual click flow: `handleClickToPick` and `handleConfirmClickToPick`
  - Dispatches pickup commands via `handlePickup`.
- **`types.ts`**: TypeScript definitions (`LogEntry`, `DetectedItem`, `DetectType`, etc.).

### Simulation Engine
- **`MujocoSim.ts`**: Central orchestrator.
  - Initializes the robot and environment model (`init`).
  - Executes physics stepping and render synchronization loop (`startLoop`).
  - Manages item picking state and speed multipliers (`pickupItems`, `setSpeedMultiplier`).
- **`RenderSystem.ts`**: Three.js scene graph manager.
  - Mesh creation from MuJoCo geoms (`GeomBuilder`).
  - Camera control and dynamic view transitions (`moveCameraTo`).
  - 2D-to-3D back-projection (`project2DTo3D`).
- **`RobotLoader.ts`**: Downloads and mounts MJCF XML models and textures into MuJoCo's in-memory virtual filesystem.

### Robotics & Kinematics
- **`IkSystem.ts`**: Coordinates target end-effector poses and joint configurations.
- **`FrankaAnalyticalIK.ts`**: Analytical geometric inverse kinematics solver for the 7-DOF Franka Panda.
- **`SequenceAnimator.ts`**: State machine orchestrating pick-and-place trajectory interpolation.

### Interaction & UI Components
- **`components/UnifiedSidebar.tsx`**: Primary interaction panel with mode dropdown (`Click to Pick`, `Gemini Robotics ER`, `Gemini 2.5 Flash`), format toggles (`Points` vs `Bounding Boxes`), and **Picked Items History**.
- **`components/ClickToPickModal.tsx`**: Interactive snapshot modal for manual point placement and bounding box drawing with animated SVG reticles.
- **`components/Toolbar.tsx`**: Floating controls for play/pause, reset, dark mode, and sidebar toggle.
- **`components/RobotSelector.tsx`**: Robot status display and coordinates.
- **`DragStateManager.ts`**: Mouse raycasting state manager.
- **`SelectionManager.ts`**: Double-click object selection and highlighting.
- **`rendering/GeomBuilder.ts`**: Factory mapping MuJoCo primitives (boxes, cylinders, spheres, capsules, meshes) into Three.js geometries.
- **`Reflector.ts`**: Mirror floor reflection rendering.
- **`CapsuleGeometry.ts`**: Custom Three.js geometry for capsule colliders.
- **`MatMath.ts`**: Matrix and vector math helpers.
- **`utils/StringUtils.ts`**: C++ null-terminated string decoder for WASM memory.
