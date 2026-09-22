/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/


import { Cpu, Sparkles } from 'lucide-react';
import { IkSolveStats, IkSolverType } from '../IkSystem';

interface RobotSelectorProps {
  gizmoStats: { pos: string, rot: string } | null;
  isDarkMode: boolean;
  ikSolver: IkSolverType;
  setIkSolver: (solver: IkSolverType) => void;
  pyrokiAvailable?: boolean;
  lastSolveStats?: IkSolveStats | null;
}

/**
 * RobotSelector
 * Overlay displaying current robot info and interactive IK Solver configuration.
 */
export function RobotSelector({
  gizmoStats,
  isDarkMode,
  ikSolver,
  setIkSolver,
  pyrokiAvailable = false,
  lastSolveStats = null
}: RobotSelectorProps) {
  const panelStyle = isDarkMode ? "bg-slate-900/80 border-white/10 text-slate-100 shadow-slate-900/20" : "bg-white/70 border-white/80 text-slate-800 shadow-slate-100/10";
  const labelStyle = isDarkMode ? "text-slate-400" : "text-slate-400";
  const valueStyle = isDarkMode ? "text-slate-300" : "text-slate-600";

  return (
    <div className="absolute top-10 left-1/2 -translate-x-1/2 min-[660px]:left-10 min-[660px]:translate-x-0 z-20 flex flex-col gap-3">
      <div className={`glass-panel px-6 py-4 rounded-3xl min-w-[260px] shadow-2xl ${panelStyle}`}>
        <div className="flex items-center justify-between gap-3 mb-3">
          <h1 className="text-lg font-bold tracking-tight leading-none">Franka Panda</h1>
          <div className="flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-slate-500/10 dark:bg-white/10 text-[10px] font-medium">
            <span
              className={`w-1.5 h-1.5 rounded-full ${
                ikSolver === 'pyroki'
                  ? (pyrokiAvailable ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400')
                  : 'bg-indigo-400'
              }`}
            />
            <span className="text-slate-600 dark:text-slate-300 font-mono text-[9px]">
              {ikSolver === 'pyroki' ? (pyrokiAvailable ? 'PyRoKi FastAPI' : 'Connecting...') : 'Analytical IK'}
            </span>
          </div>
        </div>

        {/* IK Solver Segmented Toggle */}
        <div className="flex flex-col gap-1.5">
          <div className="flex items-center justify-between text-[10px] font-semibold tracking-wider text-slate-400 uppercase">
            <span>IK Solver</span>
            <span className="text-[9px] font-normal lowercase text-slate-400">
              {ikSolver === 'pyroki' ? 'JAX optimizer' : '7-DOF closed-form'}
            </span>
          </div>

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
              title="Analytical geometric IK solver"
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
            <p className="flex justify-between gap-4"><span className={labelStyle}>POSITION:</span> <span className={`${valueStyle} font-semibold`}>{gizmoStats.pos}</span></p>
            <p className="flex justify-between gap-4"><span className={labelStyle}>ROTATION:</span> <span className={`${valueStyle} font-semibold`}>{gizmoStats.rot}</span></p>
          </div>
        </div>
      )}
    </div>
  );
}
