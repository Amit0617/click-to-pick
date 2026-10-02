/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { IkSolverType } from './IkSystem';

export interface RobotSpec {
  id: string;
  name: string;
  shortName: string;
  manufacturer: string;
  dof: number;
  menageriePath: string; // Directory in google-deepmind/mujoco_menagerie
  gripperMenageriePath?: string; // Optional gripper repository path
  sceneFile: string;
  pyrokiRobotId: string;
  defaultSolver: IkSolverType;
  availableSolvers: IkSolverType[];
  initialJoints: number[];
  homeJoints: number[];
  tcpSiteName: string;
  gripperActuatorName: string;
  gripperOpenVal: number;
  gripperCloseVal: number;
  description: string;
  badge: string;
  specs: {
    reach: string;
    payload: string;
    repeatability: string;
    weight: string;
  };
}

export const ROBOT_CONFIGS: Record<string, RobotSpec> = {
  franka_panda: {
    id: 'franka_panda',
    name: 'Franka Emika Panda',
    shortName: 'Franka Panda',
    manufacturer: 'Franka Robotics',
    dof: 7,
    menageriePath: 'franka_emika_panda',
    sceneFile: 'scene.xml',
    pyrokiRobotId: 'franka_panda',
    defaultSolver: 'pyroki',
    availableSolvers: ['pyroki', 'analytical'],
    initialJoints: [1.707, -1.754, 0.003, -2.702, 0.003, 0.951, 2.490, 0.000],
    homeJoints: [1.707, -1.754, 0.003, -2.702, 0.003, 0.951, 2.490],
    tcpSiteName: 'tcp',
    gripperActuatorName: 'gripper',
    gripperOpenVal: 255,
    gripperCloseVal: 0,
    description: '7-DOF research manipulator with torque sensors and high dynamic responsiveness.',
    badge: '7-DOF Research Arm',
    specs: {
      reach: '855 mm',
      payload: '3.0 kg',
      repeatability: '±0.1 mm',
      weight: '18 kg',
    },
  },
  ur5e: {
    id: 'ur5e',
    name: 'Universal Robots UR5e',
    shortName: 'UR5e + Robotiq 2F-85',
    manufacturer: 'Universal Robots',
    dof: 6,
    menageriePath: 'universal_robots_ur5e',
    gripperMenageriePath: 'robotiq_2f85',
    sceneFile: 'scene.xml',
    pyrokiRobotId: 'ur5e',
    defaultSolver: 'pyroki',
    availableSolvers: ['pyroki'],
    initialJoints: [1.5, -2.2708, 2.2708, -1.5708, -1.5708, 0.0],
    homeJoints: [1.5, -2.2708, 2.2708, -1.5708, -1.5708, 0.0],
    tcpSiteName: 'tcp',
    gripperActuatorName: 'gripper',
    gripperOpenVal: 0.0425,
    gripperCloseVal: 0.0,
    description: '6-DOF industrial collaborative robot integrated with Robotiq 2F-85 adaptive parallel gripper.',
    badge: '6-DOF Cobot + 2F-85',
    specs: {
      reach: '850 mm',
      payload: '5.0 kg',
      repeatability: '±0.03 mm',
      weight: '20.6 kg',
    },
  },
};

export const DEFAULT_ROBOT_ID = 'ur5e';
