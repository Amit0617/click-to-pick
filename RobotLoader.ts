/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { ROBOT_CONFIGS } from "./robots";
import { MujocoModule } from "./types";

const ROBOTIQ_MESH_FILES = [
    'base_mount.stl',
    'base.stl',
    'driver.stl',
    'coupler.stl',
    'follower.stl',
    'pad.stl',
    'silicone_pad.stl',
    'spring_link.stl'
];

/**
 * RobotLoader
 * Handles fetching robot XML files and their dependencies (meshes, textures) from remote URLs.
 * It writes these files into MuJoCo's in-memory virtual filesystem so the C++ engine can read them.
 */
export class RobotLoader {
    private mujoco: MujocoModule;

    constructor(mujocoInstance: MujocoModule) {
        this.mujoco = mujocoInstance;
    }

    /**
     * Main entry point. Downloads the main scene XML and recursively finds/downloads all included files.
     * @param robotId Target robot identifier (e.g. 'franka_panda' or 'ur5e')
     * @param sceneFile Scene XML filename (default 'scene.xml')
     * @param onProgress Optional callback to report loading progress string.
     */
    async load(robotId: string, sceneFile = 'scene.xml', onProgress?: (msg: string) => void): Promise<{ isDouble: boolean, isStacking: boolean }> {
        // 1. Clean up the virtual filesystem from previous runs
        try { this.mujoco.FS.unmount('/working'); } catch (e) { /* ignore */ }
        try { this.mujoco.FS.mkdir('/working'); } catch (e) { /* ignore */ }
        try { this.mujoco.FS.mkdir('/working/assets'); } catch (e) { /* ignore */ }

        const isDouble = false;
        const isStacking = true; // Stacking cube scene is the benchmark task environment
        
        const spec = ROBOT_CONFIGS[robotId] || ROBOT_CONFIGS.franka_panda;
        const menagerieRepo = spec.menageriePath;
        const baseUrl = `https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/main/${menagerieRepo}/`;

        // If UR5e, pre-download all Robotiq 2F-85 meshes directly into /working/assets
        // from their correct upstream repository (robotiq_2f85)
        if (spec.id === 'ur5e') {
            const robotiqBase = 'https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/main/robotiq_2f85/assets/';
            if (onProgress) onProgress('Loading Robotiq 2F-85 gripper meshes...');
            
            await Promise.all(ROBOTIQ_MESH_FILES.map(async (meshFile) => {
                try {
                    const res = await fetch(robotiqBase + meshFile);
                    if (res.ok) {
                        const buffer = new Uint8Array(await res.arrayBuffer());
                        this.mujoco.FS.writeFile(`/working/assets/${meshFile}`, buffer);
                    } else {
                        console.warn(`Could not load Robotiq mesh ${meshFile}: ${res.status}`);
                    }
                } catch (e) {
                    console.warn(`Network error downloading Robotiq mesh ${meshFile}:`, e);
                }
            }));
        }

        const downloaded = new Set<string>();
        const queue: Array<{ path: string; customBaseUrl?: string }> = [];
        const parser = new DOMParser();

        queue.push({ path: sceneFile });

        // Process queue until all dependencies are downloaded
        while (queue.length > 0) {
            const item = queue.shift()!;
            const fname = item.path;
            const currentBase = item.customBaseUrl || baseUrl;
            const key = `${currentBase}${fname}`;
            
            if (downloaded.has(key)) continue;
            downloaded.add(key);

            if (onProgress) {
                const displayName = fname.split('/').pop() || fname;
                onProgress(`Loading ${displayName}...`);
            }

            // Fetch file from network
            let res: Response;
            try {
                res = await fetch(currentBase + fname);
                if (!res.ok) {
                    console.warn(`Failed to fetch ${fname} from ${currentBase}: ${res.status}`);
                    continue;
                }
            } catch (err) {
                console.warn(`Network error fetching ${fname}:`, err);
                continue;
            }

            // Ensure virtual directory structure exists (e.g., /working/assets/)
            const dirParts = fname.split('/');
            dirParts.pop(); // remove filename
            let currentPath = '/working';
            for (const part of dirParts) {
                currentPath += '/' + part;
                try { this.mujoco.FS.mkdir(currentPath); } catch (e) { /* ignore */ }
            }

            // If it's an XML, patch it and scan for more dependencies
            if (fname.endsWith('.xml')) {
                let text = await res.text();
                text = this.patchRobotXml(fname, sceneFile, spec.id, text);
                
                // Write text file to virtual FS
                this.mujoco.FS.writeFile(`/working/${fname}`, text);
                // Scan for <include file="...">, <mesh file="...">, etc.
                this.scanDependencies(text, fname, currentBase, parser, downloaded, queue);
            } else {
                // Binary files (STL, OBJ, PNG) get written directly
                const buffer = new Uint8Array(await res.arrayBuffer());
                this.mujoco.FS.writeFile(`/working/${fname}`, buffer);
            }
        }
        return { isDouble, isStacking };
    }

    /**
     * Modifies standard XMLs to add demo objects (cubes, tray) and configure TCP sites/actuators.
     */
    private patchRobotXml(fname: string, sceneFile: string, robotId: string, text: string): string {
        // 1. Inject Table, Tray & Stacking Cubes into scene.xml for the unified picking environment
        if (fname === sceneFile) {
            let injection = '';
            const colors = [
                '0.8 0.1 0.1 1', // Red
                '0.0 0.8 0.8 1', // Cyan
                '0.1 0.8 0.1 1', // Green
                '0.8 0.8 0.1 1'  // Yellow
            ];
            
            const positions: { x: number; y: number }[] = [];
            
            // Inject 20 cubes for pick & stack demo
            for (let i = 0; i < 20; i++) {
                let x = 0;
                let y = 0;
                let valid = false;
                let attempts = 0;
                
                while (!valid && attempts < 200) {
                    const minR = 0.35;
                    const maxR = 0.70;
                    const r = Math.sqrt(Math.random() * (maxR*maxR - minR*minR) + minR*minR);
                    const theta = Math.random() * 2 * Math.PI;
                    
                    x = r * Math.cos(theta);
                    y = r * Math.sin(theta);
                    
                    valid = true;
                    
                    // Avoid tray base (0.55, 0)
                    const distStack = Math.sqrt((x - 0.55)**2 + (y - 0)**2);
                    if (distStack < 0.32) valid = false;

                    // Avoid robot base center
                    if (Math.sqrt(x*x + y*y) < 0.28) valid = false;

                    // Avoid other cubes
                    if (valid) {
                        for (const p of positions) {
                            const d2 = (p.x - x)**2 + (p.y - y)**2;
                            if (d2 < 0.004) {
                                valid = false; 
                                break; 
                            }
                        }
                    }
                    attempts++;
                }

                if (valid) {
                    positions.push({ x, y });
                    const color = colors[i % 4];
                    injection += `<body name="cube${i}" pos="${x.toFixed(3)} ${y.toFixed(3)} 0.02"><freejoint/><geom type="box" size="0.02 0.02 0.02" rgba="${color}" mass="0.05" friction="1.5 0.3 0.1" solref="0.01 1" solimp="0.95 0.99 0.001 0.5 2" condim="4"/></body>`;
                }
            }
            
            // Stacking target tray (20cm x 20cm) placed in comfortable reach of both manipulators
            injection += `<body name="stack_base" pos="0.55 0 0.0"><geom type="box" size="0.1 0.1 0.005" rgba="0.25 0.25 0.25 1"/></body>`;
            text = text.replace('</worldbody>', injection + '</worldbody>');
        }

        // 2. Franka Panda specific patches: ensure TCP site and named gripper actuator
        if (robotId === 'franka_panda' && fname.endsWith('panda.xml')) {
            text = text
                .replace(/(<body[^>]*name=["']hand["'][^>]*>)/, '$1<site name="tcp" pos="0 0 0.10" size="0.01" rgba="1 0 0 0.5" group="1"/>')
                .replace(/name=["']actuator8["']/, 'name="gripper"');
        }

        // 3. Universal Robots UR5e specific patches: align base to forward (+X) and attach parallel Robotiq 2F-85 gripper
        if (robotId === 'ur5e' && fname.endsWith('ur5e.xml')) {
            // Align base with standard world forward coordinates (+X forward, matching ROS / PyRoKi)
            text = text.replace('<body name="base" quat="0 0 0 -1"', '<body name="base" quat="1 0 0 0"');

            // Include Robotiq meshes & materials into asset
            // Note: Robotiq Menagerie STL meshes are in millimeters, so scale="0.001 0.001 0.001" is mandatory
            const robotiqAssets = `
    <material name="metal" rgba="0.58 0.58 0.58 1"/>
    <material name="silicone" rgba="0.1882 0.1882 0.1882 1"/>
    <material name="gray" rgba="0.4627 0.4627 0.4627 1"/>
    <mesh file="base_mount.stl" scale="0.001 0.001 0.001"/>
    <mesh file="base.stl" scale="0.001 0.001 0.001"/>
    <mesh file="driver.stl" scale="0.001 0.001 0.001"/>
    <mesh file="coupler.stl" scale="0.001 0.001 0.001"/>
    <mesh file="follower.stl" scale="0.001 0.001 0.001"/>
    <mesh file="pad.stl" scale="0.001 0.001 0.001"/>
    <mesh file="silicone_pad.stl" scale="0.001 0.001 0.001"/>
    <mesh file="spring_link.stl" scale="0.001 0.001 0.001"/>
            `;
            text = text.replace('</asset>', robotiqAssets + '</asset>');

            // Attach Robotiq gripper geometry and TCP site onto wrist_3_link
            // Prismatic slide joints provide authentic horizontal parallel motion (85 mm stroke)
            const gripperMount = `
                  <body name="robotiq_mount" pos="0 0.1 0" quat="-1 1 0 0">
                    <geom mesh="base_mount" class="visual" material="black"/>
                    <body name="robotiq_base" pos="0 0 0.0038" quat="1 0 0 -1">
                      <geom mesh="base" class="visual" material="black"/>
                      <body name="left_finger" pos="0 -0.012 0.06">
                        <joint name="finger_joint1" type="slide" axis="0 -1 0" range="0 0.0425" damping="20"/>
                        <geom type="box" size="0.012 0.006 0.04" pos="0 -0.006 0.02" rgba="0.25 0.25 0.25 1" mass="0.05"/>
                        <geom type="box" size="0.011 0.004 0.025" pos="0 0.002 0.055" rgba="0.15 0.15 0.15 1" friction="2.0 0.1 0.001" solref="0.01 1" solimp="0.95 0.99 0.001" condim="4" mass="0.02"/>
                      </body>
                      <body name="right_finger" pos="0 0.012 0.06">
                        <joint name="finger_joint2" type="slide" axis="0 1 0" range="0 0.0425" damping="20"/>
                        <geom type="box" size="0.012 0.006 0.04" pos="0 0.006 0.02" rgba="0.25 0.25 0.25 1" mass="0.05"/>
                        <geom type="box" size="0.011 0.004 0.025" pos="0 -0.002 0.055" rgba="0.15 0.15 0.15 1" friction="2.0 0.1 0.001" solref="0.01 1" solimp="0.95 0.99 0.001" condim="4" mass="0.02"/>
                      </body>
                      <!-- Dedicated TCP center site between gripper finger pads for IK tracking -->
                      <site name="tcp" pos="0 0 0.135" size="0.01" rgba="1 0 0 0.5" group="1"/>
                    </body>
                  </body>
            `;
            
            // Insert gripper mount into wrist_3_link
            text = text.replace(/(<site name="attachment_site"[^>]*\/>)/, '$1\n' + gripperMount);

            // Couple both fingers symmetrically via equality constraint
            const equality = `
    <equality>
      <joint joint1="finger_joint1" joint2="finger_joint2" polycoef="0 1 0 0 0"/>
    </equality>
            `;
            text = text.replace('</mujoco>', equality + '</mujoco>');

            // Coupled position actuator driving both symmetric fingers horizontally
            const gripperActuator = `
    <position name="gripper" joint="finger_joint1" ctrlrange="0 0.0425" kp="800" kv="50" forcerange="-100 100"/>
            `;
            text = text.replace('</actuator>', gripperActuator + '</actuator>');
        }

        return text;
    }

    // Finds all files referenced in the XML so we can download them too
    private scanDependencies(
        xmlString: string, 
        currentFile: string, 
        currentBaseUrl: string, 
        parser: DOMParser, 
        downloaded: Set<string>, 
        queue: Array<{ path: string; customBaseUrl?: string }>
    ) {
        const xmlDoc = parser.parseFromString(xmlString, 'text/xml');
        
        // Check if the XML defines specific directories for assets
        const compiler = xmlDoc.querySelector('compiler');
        const meshDir = compiler?.getAttribute('meshdir') || '';
        const textureDir = compiler?.getAttribute('texturedir') || '';
        
        // Calculate relative path of current file
        const currentDir = currentFile.includes('/') ? currentFile.substring(0, currentFile.lastIndexOf('/') + 1) : '';

        // Find all elements with a 'file' attribute
        xmlDoc.querySelectorAll('[file]').forEach(el => {
            const fileAttr = el.getAttribute('file');
            if (!fileAttr) return;

            // Robotiq meshes are pre-downloaded from their dedicated repository into /working/assets
            // Do not attempt to re-fetch them from universal_robots_ur5e
            if (ROBOTIQ_MESH_FILES.includes(fileAttr)) {
                return;
            }
            
            // Prepend appropriate directory based on tag type
            let prefix = '';
            if (el.tagName.toLowerCase() === 'mesh') {
                prefix = meshDir ? meshDir + '/' : '';
            } else if (['texture', 'hfield'].includes(el.tagName.toLowerCase())) {
                prefix = textureDir ? textureDir + '/' : '';
            }
            
            // Normalize path (resolve '..' and '.')
            let fullPath = (currentDir + prefix + fileAttr).replace(/\/\//g, '/');
            const parts = fullPath.split('/');
            const norm: string[] = [];
            for (const p of parts) { if (p === '..') norm.pop(); else if (p !== '.') norm.push(p); }
            fullPath = norm.join('/');
            
            const key = `${currentBaseUrl}${fullPath}`;
            if (!downloaded.has(key)) {
                queue.push({ path: fullPath, customBaseUrl: currentBaseUrl });
            }
        });
    }
}
