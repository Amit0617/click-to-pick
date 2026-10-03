/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { ROBOT_CONFIGS, RobotSpec } from "./robots";
import { MujocoModule } from "./types";

/**
 * Helper to get direct child elements with optional tagName filter
 */
function getDirectElements(el: Element, tagName?: string): Element[] {
    const res: Element[] = [];
    if (!el || !el.childNodes) return res;
    for (let i = 0; i < el.childNodes.length; i++) {
        const node = el.childNodes[i];
        if (node.nodeType === 1) { // ELEMENT_NODE
            const elem = node as Element;
            if (!tagName || elem.tagName.toLowerCase() === tagName.toLowerCase()) {
                res.push(elem);
            }
        }
    }
    return res;
}

/**
 * Helper to find descendant element matching tag and optional attribute
 */
function findElement(root: Document | Element, tagName: string, attrName?: string, attrVal?: string): Element | null {
    if (!root) return null;
    const elements = root.getElementsByTagName(tagName);
    for (let i = 0; i < elements.length; i++) {
        const el = elements[i];
        if (!attrName || el.getAttribute(attrName) === attrVal) {
            return el;
        }
    }
    return null;
}

/**
 * RobotLoader
 * Handles dynamically fetching robot and gripper XML files and their dependencies (meshes, textures)
 * from remote repositories at loading time, attaching grippers to manipulators via structured DOM logic,
 * and populating MuJoCo's in-memory virtual filesystem.
 */
export class RobotLoader {
    private mujoco: MujocoModule;
    private cachedGripperXml: string | null = null;
    private downloadedGripperAssets = new Set<string>();

    constructor(mujocoInstance: MujocoModule) {
        this.mujoco = mujocoInstance;
    }

    /**
     * Main entry point. Downloads the main scene XML and recursively finds/downloads all dependencies.
     * If a gripper is specified, downloads its XML and referenced assets at loading time and attaches it.
     */
    async load(robotId: string, sceneFile = 'scene.xml', onProgress?: (msg: string) => void): Promise<{ isDouble: boolean, isStacking: boolean }> {
        // 1. Clean up virtual filesystem from previous runs
        try { this.mujoco.FS.unmount('/working'); } catch (e) { /* ignore */ }
        try { this.mujoco.FS.mkdir('/working'); } catch (e) { /* ignore */ }
        try { this.mujoco.FS.mkdir('/working/assets'); } catch (e) { /* ignore */ }

        const isDouble = false;
        const isStacking = true; // Stacking cube scene is the benchmark task environment
        
        const spec = ROBOT_CONFIGS[robotId] || ROBOT_CONFIGS.franka_panda;
        const menagerieRepo = spec.menageriePath;
        const baseUrl = `https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/main/${menagerieRepo}/`;

        // 2. If robot config specifies a gripper, dynamically fetch gripper XML & its assets at loading time
        this.cachedGripperXml = null;
        this.downloadedGripperAssets.clear();
        if (spec.gripperMenageriePath) {
            this.cachedGripperXml = await this.downloadGripperAssets(spec, onProgress);
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
                text = this.patchRobotXml(fname, sceneFile, spec, text);
                
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
     * Downloads the gripper XML and all assets referenced by it at loading time.
     */
    private async downloadGripperAssets(spec: RobotSpec, onProgress?: (msg: string) => void): Promise<string | null> {
        if (!spec.gripperMenageriePath) return null;
        const repo = spec.gripperMenageriePath;
        const xmlFile = spec.gripperXmlFile || '2f85.xml';
        const gripperXmlUrl = `https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/main/${repo}/${xmlFile}`;

        if (onProgress) onProgress(`Fetching ${xmlFile}...`);
        const res = await fetch(gripperXmlUrl);
        if (!res.ok) {
            console.warn(`Failed to fetch gripper XML from ${gripperXmlUrl}: ${res.status}`);
            return null;
        }

        const gripperXmlText = await res.text();
        const parser = new DOMParser();
        const gripperDoc = parser.parseFromString(gripperXmlText, 'text/xml');

        // Dynamically find all asset files (meshes, textures) referenced by the gripper XML
        const assetFiles = new Set<string>();
        gripperDoc.querySelectorAll('[file]').forEach(el => {
            const f = el.getAttribute('file');
            if (f) assetFiles.add(f);
        });

        if (onProgress && assetFiles.size > 0) {
            onProgress(`Loading ${spec.shortName} gripper assets (${assetFiles.size} meshes)...`);
        }

        const assetsBaseUrl = `https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/main/${repo}/assets/`;

        await Promise.all(Array.from(assetFiles).map(async (meshFile) => {
            try {
                const r = await fetch(assetsBaseUrl + meshFile);
                if (r.ok) {
                    const buffer = new Uint8Array(await r.arrayBuffer());
                    this.mujoco.FS.writeFile(`/working/assets/${meshFile}`, buffer);
                    this.downloadedGripperAssets.add(meshFile);
                } else {
                    console.warn(`Could not load gripper mesh ${meshFile}: ${r.status}`);
                }
            } catch (e) {
                console.warn(`Network error downloading gripper asset ${meshFile}:`, e);
            }
        }));

        return gripperXmlText;
    }

    /**
     * Modifies standard XMLs to add demo objects (cubes, tray) and configure TCP sites/actuators.
     * When a gripper is cached, attaches it to the manipulator's attachment site.
     */
    private patchRobotXml(fname: string, sceneFile: string, spec: RobotSpec, text: string): string {
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
        if (spec.id === 'franka_panda' && fname.endsWith('panda.xml')) {
            text = text
                .replace(/(<body[^>]*name=["']hand["'][^>]*>)/, '$1<site name="tcp" pos="0 0 0.10" size="0.01" rgba="1 0 0 0.5" group="1"/>')
                .replace(/name=["']actuator8["']/, 'name="gripper"');
        }

        // 3. Attach Gripper to Manipulator (e.g. Robotiq 2F-85 onto Universal Robots UR5e)
        if (spec.gripperMenageriePath && this.cachedGripperXml && fname.endsWith(spec.id + '.xml')) {
            // Ensure base coordinates align with standard forward (+X forward, matching ROS / PyRoKi)
            if (spec.id === 'ur5e') {
                text = text.replace('<body name="base" quat="0 0 0 -1"', '<body name="base" quat="1 0 0 0"');
            }

            // Cleanly attach gripper to manipulator's attachment site using structured DOM operations
            text = this.attachGripperToManipulator(
                text, 
                this.cachedGripperXml, 
                spec.attachmentSiteName || 'attachment_site'
            );
        }

        return text;
    }

    /**
     * Attaches an independently defined gripper model onto the manipulator's attachment site
     * using standard DOM tree operations to merge defaults, assets, worldbody bodies, contacts,
     * tendons, equality constraints, and actuators.
     */
    private attachGripperToManipulator(robotXmlStr: string, gripperXmlStr: string, attachmentSiteName = "attachment_site"): string {
        const parser = new DOMParser();
        const serializer = new XMLSerializer();

        const robotDoc = parser.parseFromString(robotXmlStr, "text/xml");
        const gripperDoc = parser.parseFromString(gripperXmlStr, "text/xml");

        const robotMujoco = robotDoc.getElementsByTagName("mujoco")[0];
        const gripperMujoco = gripperDoc.getElementsByTagName("mujoco")[0];

        if (!robotMujoco || !gripperMujoco) {
            throw new Error("Invalid MJCF XML structure");
        }

        // Ensure class names don't collide with existing robot classes (e.g. 'visual', 'collision')
        const existingClasses = new Set<string>();
        const robotDefaultTags = robotDoc.getElementsByTagName("default");
        for (let i = 0; i < robotDefaultTags.length; i++) {
            const cls = robotDefaultTags[i].getAttribute("class");
            if (cls) existingClasses.add(cls);
        }

        // If 'visual' already exists in the robot model, prefix gripper's visual class
        if (existingClasses.has("visual")) {
            const gripperVisuals = gripperDoc.getElementsByTagName("default");
            for (let i = 0; i < gripperVisuals.length; i++) {
                if (gripperVisuals[i].getAttribute("class") === "visual") {
                    gripperVisuals[i].setAttribute("class", "2f85_visual");
                }
            }
            const geoms = gripperDoc.getElementsByTagName("geom");
            for (let i = 0; i < geoms.length; i++) {
                if (geoms[i].getAttribute("class") === "visual") {
                    geoms[i].setAttribute("class", "2f85_visual");
                }
            }
        }

        // If 'collision' already exists in the robot model, prefix gripper's collision class
        if (existingClasses.has("collision")) {
            const gripperCols = gripperDoc.getElementsByTagName("default");
            for (let i = 0; i < gripperCols.length; i++) {
                if (gripperCols[i].getAttribute("class") === "collision") {
                    gripperCols[i].setAttribute("class", "2f85_collision");
                }
            }
            const geoms = gripperDoc.getElementsByTagName("geom");
            for (let i = 0; i < geoms.length; i++) {
                if (geoms[i].getAttribute("class") === "collision") {
                    geoms[i].setAttribute("class", "2f85_collision");
                }
            }
        }

        // 1. Merge <default>
        let robotDefault = getDirectElements(robotMujoco, "default")[0];
        if (!robotDefault) {
            robotDefault = robotDoc.createElement("default");
            robotMujoco.insertBefore(robotDefault, robotMujoco.firstChild);
        }
        const gripperDefault = getDirectElements(gripperMujoco, "default")[0];
        if (gripperDefault) {
            const defChildren = getDirectElements(gripperDefault);
            defChildren.forEach(child => {
                robotDefault.appendChild(robotDoc.importNode(child, true));
            });
        }

        // 2. Merge <asset>
        let robotAsset = getDirectElements(robotMujoco, "asset")[0];
        if (!robotAsset) {
            robotAsset = robotDoc.createElement("asset");
            robotMujoco.appendChild(robotAsset);
        }
        const gripperAsset = getDirectElements(gripperMujoco, "asset")[0];
        if (gripperAsset) {
            const assetChildren = getDirectElements(gripperAsset);
            assetChildren.forEach(child => {
                const name = child.getAttribute("name");
                const file = child.getAttribute("file");
                if (name && findElement(robotAsset, child.tagName, "name", name)) return;
                if (file && findElement(robotAsset, child.tagName, "file", file)) return;
                robotAsset.appendChild(robotDoc.importNode(child, true));
            });
        }

        // 3. Find attachment site in robot worldbody
        const attachmentSite = findElement(robotDoc, "site", "name", attachmentSiteName) || findElement(robotDoc, "site", "name", "tcp");
        if (!attachmentSite || !attachmentSite.parentNode) {
            throw new Error(`Attachment site '${attachmentSiteName}' not found in robot model`);
        }

        const parentBody = attachmentSite.parentNode as Element;
        const sitePos = attachmentSite.getAttribute("pos") || "0 0 0";
        const siteQuat = attachmentSite.getAttribute("quat") || "1 0 0 0";

        // 4. Attach Gripper root body from gripper worldbody
        const gripperWorldbody = getDirectElements(gripperMujoco, "worldbody")[0];
        if (!gripperWorldbody) {
            throw new Error("No worldbody found in gripper XML");
        }
        const gripperRootBodies = getDirectElements(gripperWorldbody, "body");
        if (gripperRootBodies.length === 0) {
            throw new Error("No root body found in gripper worldbody");
        }
        const gripperRootBody = gripperRootBodies[0];
        const importedGripperBody = robotDoc.importNode(gripperRootBody, true) as Element;
        
        // Position gripper at attachment site
        importedGripperBody.setAttribute("pos", sitePos);
        importedGripperBody.setAttribute("quat", siteQuat);

        // Rename internal "base" body to "robotiq_base" to avoid collision with robot pedestal "base"
        const internalBodies = importedGripperBody.getElementsByTagName("body");
        for (let i = 0; i < internalBodies.length; i++) {
            if (internalBodies[i].getAttribute("name") === "base") {
                internalBodies[i].setAttribute("name", "robotiq_base");
            }
        }
        if (importedGripperBody.getAttribute("name") === "base") {
            importedGripperBody.setAttribute("name", "robotiq_base");
        }

        // Add dedicated TCP site if not present for inverse kinematics tracking
        if (!findElement(importedGripperBody, "site", "name", "tcp")) {
            const tcpSite = robotDoc.createElement("site");
            tcpSite.setAttribute("name", "tcp");
            tcpSite.setAttribute("pos", "0 0 0.145"); // as per Robotiq 2F-85 in MuJoCo menagerie
            tcpSite.setAttribute("size", "0.01");
            tcpSite.setAttribute("rgba", "1 0 0 0.5");
            tcpSite.setAttribute("group", "1");
            importedGripperBody.appendChild(tcpSite);
        }

        parentBody.appendChild(importedGripperBody);

        // Helper to insert section before <actuator> or at end of <mujoco>
        const insertBeforeActuator = (newSection: Element) => {
            const actuator = getDirectElements(robotMujoco, "actuator")[0];
            if (actuator) {
                robotMujoco.insertBefore(newSection, actuator);
            } else {
                robotMujoco.appendChild(newSection);
            }
        };

        // 5. Merge <contact>
        const gripperContact = getDirectElements(gripperMujoco, "contact")[0];
        if (gripperContact) {
            let robotContact = getDirectElements(robotMujoco, "contact")[0];
            if (!robotContact) {
                robotContact = robotDoc.createElement("contact");
                insertBeforeActuator(robotContact);
            }
            const contactChildren = getDirectElements(gripperContact);
            contactChildren.forEach(child => {
                const imp = robotDoc.importNode(child, true) as Element;
                if (imp.getAttribute("body1") === "base") imp.setAttribute("body1", "robotiq_base");
                if (imp.getAttribute("body2") === "base") imp.setAttribute("body2", "robotiq_base");
                robotContact.appendChild(imp);
            });
        }

        // 6. Merge <tendon>
        const gripperTendon = getDirectElements(gripperMujoco, "tendon")[0];
        if (gripperTendon) {
            let robotTendon = getDirectElements(robotMujoco, "tendon")[0];
            if (!robotTendon) {
                robotTendon = robotDoc.createElement("tendon");
                insertBeforeActuator(robotTendon);
            }
            const tendonChildren = getDirectElements(gripperTendon);
            tendonChildren.forEach(child => {
                robotTendon.appendChild(robotDoc.importNode(child, true));
            });
        }

        // 7. Merge <equality>
        const gripperEquality = getDirectElements(gripperMujoco, "equality")[0];
        if (gripperEquality) {
            let robotEquality = getDirectElements(robotMujoco, "equality")[0];
            if (!robotEquality) {
                robotEquality = robotDoc.createElement("equality");
                insertBeforeActuator(robotEquality);
            }
            const eqChildren = getDirectElements(gripperEquality);
            eqChildren.forEach(child => {
                robotEquality.appendChild(robotDoc.importNode(child, true));
            });
        }

        // 8. Merge <actuator>
        const gripperActuator = getDirectElements(gripperMujoco, "actuator")[0];
        if (gripperActuator) {
            let robotActuator = getDirectElements(robotMujoco, "actuator")[0];
            if (!robotActuator) {
                robotActuator = robotDoc.createElement("actuator");
                robotMujoco.appendChild(robotActuator);
            }
            const actChildren = getDirectElements(gripperActuator);
            actChildren.forEach(act => {
                const imp = robotDoc.importNode(act, true) as Element;
                // Standardize actuator name to 'gripper'
                imp.setAttribute("name", "gripper");
                // Standardize control: 255 = fully open (0.0 rad), 0 = fully closed (0.8 rad)
                imp.setAttribute("ctrlrange", "0 255");
                imp.setAttribute("gainprm", "-0.3137255 0 0");
                imp.setAttribute("biasprm", "80 -100 -10");
                imp.setAttribute("forcerange", "-100 100");
                robotActuator.appendChild(imp);
            });
        }

        return serializer.serializeToString(robotDoc);
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

            // Gripper meshes downloaded dynamically into /working/assets
            if (this.downloadedGripperAssets.has(fileAttr)) {
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
