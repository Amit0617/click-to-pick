/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { Cpu, Sparkles, CheckCircle2 } from 'lucide-react';
import { IkSolveStats, IkSolverType } from '../IkSystem';
import { ROBOT_CONFIGS, RobotSpec } from '../robots';

interface RobotSelectorProps {
  currentRobotId?: string;
  gizmoStats: { pos: string, rot: string } | null;
  isDarkMode: boolean;
  ikSolver: IkSolverType;
  setIkSolver: (solver: IkSolverType) => void;
  pyrokiAvailable?: boolean;
  loadedRobots?: string[];
  lastSolveStats?: IkSolveStats | null;
}

/**
 * RobotSelector
 * Overlay displaying active robot kinematic status, gripper specs, and interactive IK Solver mode.
 */
export function RobotSelector({
  currentRobotId = 'franka_panda',
  gizmoStats,
  isDarkMode,
  ikSolver,
  setIkSolver,
  pyrokiAvailable = false,
  loadedRobots = [],
  lastSolveStats = null,
}: RobotSelectorProps) {
  const robot: RobotSpec = ROBOT_CONFIGS[currentRobotId] || ROBOT_CONFIGS.franka_panda;
  const supportsAnalytical = robot.availableSolvers.includes('analytical');

  const panelStyle = isDarkMode
    ? 'bg-slate-900/80 border-white/10 text-slate-100 shadow-slate-900/20'
    : 'bg-white/70 border-white/80 text-slate-800 shadow-slate-100/10';
  const labelStyle = 'text-slate-400';
  const valueStyle = isDarkMode ? 'text-slate-300' : 'text-slate-600';

  const isWarmedUp = loadedRobots.includes(robot.pyrokiRobotId);

  return (
    <div className="absolute top-20 min-[660px]:top-10 left-1/2 -translate-x-1/2 min-[660px]:left-10 min-[660px]:translate-x-0 z-20 flex flex-col gap-3 pointer-events-auto">
      <div className={`glass-panel px-6 py-4 rounded-3xl min-w-[280px] shadow-2xl ${panelStyle}`}>
        <div className="flex items-center justify-between gap-3 mb-2">
          <div>
            <h1 className="text-base font-bold tracking-tight leading-tight">{robot.shortName}</h1>
            <p className="text-[10px] text-slate-400 font-medium">{robot.manufacturer}</p>
          </div>
          <div className="flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-slate-500/10 dark:bg-white/10 text-[10px] font-medium">
            <span
              className={`w-1.5 h-1.5 rounded-full ${
                ikSolver === 'pyroki'
                  ? pyrokiAvailable
                    ? isWarmedUp
                      ? 'bg-emerald-400 animate-pulse'
                      : 'bg-amber-400 animate-pulse'
                    : 'bg-slate-400'
                  : 'bg-indigo-400'
              }`}
            />
            <span className="text-slate-600 dark:text-slate-300 font-mono text-[9px]">
              {ikSolver === 'pyroki'
                ? pyrokiAvailable
                  ? isWarmedUp
                    ? 'PyRoKi Active'
                    : 'PyRoKi Ready'
                  : 'Connecting...'
                : 'Analytical IK'}
            </span>
          </div>
        </div>

        {/* IK Solver Configuration */}
        <div className="flex flex-col gap-1.5 mt-3">
          <div className="flex items-center justify-between text-[10px] font-semibold tracking-wider text-slate-400 uppercase">
            <span>IK Solver</span>
            <span className="text-[9px] font-normal lowercase text-slate-400">
              {ikSolver === 'pyroki' ? 'JAX optimizer' : 'closed-form geometric'}
            </span>
          </div>

          {supportsAnalytical ? (
            <div className="grid grid-cols-2 p-1 bg-slate-200/50 dark:bg-slate-800/80 rounded-xl border border-black/5 dark:border-white/5">
              <button
                id="ik-solver-analytical-btn"
                type="button"
                onClick={() => setIkSolver('analytical')}
                className={`flex items-center justify-center gap-1.5 py-1.5 px-2 rounded-lg text-xs font-medium transition-all ${
                  ikSolver === 'analytical'
                    ? 'bg-white dark:bg-slate-700 text-slate-900 dark:text-white shadow-sm font-semibold'
                    : 'text-slate-500 hover:text-slate-800 dark:text-slate-400 dark:hover:text-slate-200'
                }`}
                title="Analytical closed-form IK solver"
              >
                <Cpu className="w-3.5 h-3.5 opacity-80" />
                <span>Analytical</span>
              </button>

              <button
                id="ik-solver-pyroki-btn"
                type="button"
                onClick={() => setIkSolver('pyroki')}
                className={`flex items-center justify-center gap-1.5 py-1.5 px-2 rounded-lg text-xs font-medium transition-all ${
                  ikSolver === 'pyroki'
                    ? 'bg-white dark:bg-slate-700 text-slate-900 dark:text-white shadow-sm font-semibold'
                    : 'text-slate-500 hover:text-slate-800 dark:text-slate-400 dark:hover:text-slate-200'
                }`}
                title="PyRoKi JAX differentiable optimization solver via FastAPI"
              >
                <Sparkles className="w-3.5 h-3.5 opacity-80 text-amber-500 dark:text-amber-400" />
                <span>PyRoKi</span>
              </button>
            </div>
          ) : (
            <div className="p-2 rounded-xl bg-indigo-500/10 border border-indigo-500/20 text-indigo-700 dark:text-indigo-300 flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-indigo-500 shrink-0" />
              <div className="text-[11px] leading-tight">
                <span className="font-bold">PyRoKi JAX Active</span>
                <p className="text-[9px] text-slate-500 dark:text-slate-400">Generalized 6-DOF least-squares optimizer</p>
              </div>
            </div>
          )}

          {lastSolveStats && (
            <div className="text-[9px] text-center text-slate-400 font-mono mt-0.5">
              Last solve:{' '}
              <span className="text-slate-600 dark:text-slate-300 font-semibold capitalize">
                {lastSolveStats.method}
              </span>{' '}
              in{' '}
              <span className="text-emerald-500 dark:text-emerald-400 font-bold">
                {lastSolveStats.timeMs.toFixed(1)}ms
              </span>
            </div>
          )}
        </div>
      </div>
      
      {gizmoStats && (
        <div className={`glass-card px-5 py-3 rounded-2xl flex flex-col gap-2 shadow-sm ${isDarkMode ? 'bg-slate-800/60 border-white/5' : 'bg-white/40 border-white/50'}`}>
          <div className="font-mono text-[9px] space-y-0.5">
            <p className="flex justify-between gap-4"><span className={labelStyle}>TCP POSITION:</span> <span className={`${valueStyle} font-semibold`}>{gizmoStats.pos}</span></p>
            <p className="flex justify-between gap-4"><span className={labelStyle}>TCP ROTATION:</span> <span className={`${valueStyle} font-semibold`}>{gizmoStats.rot}</span></p>
          </div>
        </div>
      )}
    </div>
  );
}
