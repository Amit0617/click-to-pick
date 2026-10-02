/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useRef, useEffect } from 'react';
import { Bot, ChevronDown, Check, Sparkles, Cpu, Layers } from 'lucide-react';
import { ROBOT_CONFIGS, RobotSpec } from '../robots';

interface RobotSelectorPillProps {
  currentRobotId: string;
  onSelectRobot: (robotId: string) => void;
  isDarkMode: boolean;
  pyrokiAvailable?: boolean;
  isLoading?: boolean;
}

/**
 * RobotSelectorPill
 * Floating glass dropdown positioned at the top-center of the viewport,
 * enabling immediate switching between robotic manipulators (Franka Panda, UR5e).
 */
export function RobotSelectorPill({
  currentRobotId,
  onSelectRobot,
  isDarkMode,
  pyrokiAvailable = false,
  isLoading = false,
}: RobotSelectorPillProps) {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const activeRobot: RobotSpec = ROBOT_CONFIGS[currentRobotId] || ROBOT_CONFIGS.franka_panda;
  const robotList = Object.values(ROBOT_CONFIGS);

  // Close dropdown on outside click or ESC key
  useEffect(() => {
    const handleOutsideClick = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setIsOpen(false);
      }
    };
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setIsOpen(false);
    };

    if (isOpen) {
      document.addEventListener('mousedown', handleOutsideClick);
      document.addEventListener('keydown', handleKeyDown);
    }
    return () => {
      document.removeEventListener('mousedown', handleOutsideClick);
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [isOpen]);

  const pillTheme = isDarkMode
    ? 'bg-slate-900/80 border-white/15 text-slate-100 shadow-slate-950/40 hover:bg-slate-900/95'
    : 'bg-white/85 border-white/90 text-slate-800 shadow-slate-200/50 hover:bg-white';

  const menuTheme = isDarkMode
    ? 'bg-slate-900/95 border-white/10 text-slate-100 shadow-2xl shadow-black/60 divide-white/5'
    : 'bg-white/95 border-slate-200/80 text-slate-800 shadow-2xl shadow-slate-200/60 divide-slate-100';

  return (
    <div
      ref={dropdownRef}
      className="absolute top-5 left-1/2 -translate-x-1/2 z-30 flex flex-col items-center"
    >
      {/* Main Pill Button */}
      <button
        type="button"
        id="robot-selector-top-pill"
        onClick={() => !isLoading && setIsOpen(!isOpen)}
        disabled={isLoading}
        aria-haspopup="listbox"
        aria-expanded={isOpen}
        className={`glass-panel px-4 py-2 rounded-full flex items-center gap-3 transition-all duration-200 border cursor-pointer select-none active:scale-[0.98] ${pillTheme} ${
          isOpen ? 'ring-2 ring-indigo-500/40 shadow-lg' : ''
        } ${isLoading ? 'opacity-80 cursor-wait' : ''}`}
        title="Switch active robot manipulator morphology"
      >
        <div className="flex items-center gap-2">
          <div className="w-6 h-6 rounded-full bg-indigo-500/15 dark:bg-indigo-400/20 text-indigo-600 dark:text-indigo-400 flex items-center justify-center shrink-0">
            <Bot className="w-3.5 h-3.5" />
          </div>
          <div className="flex items-baseline gap-1.5 text-left">
            <span className="text-xs font-bold tracking-tight whitespace-nowrap">
              {activeRobot.name}
            </span>
            <span className="text-[10px] text-slate-400 dark:text-slate-500 font-mono">
              · {activeRobot.dof}-DOF
            </span>
          </div>
        </div>

        {/* Status Indicator */}
        <div className="flex items-center gap-1.5 pl-1 pr-0.5 border-l border-black/5 dark:border-white/10">
          <span
            className={`w-1.5 h-1.5 rounded-full ${
              pyrokiAvailable ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'
            }`}
            title={pyrokiAvailable ? 'PyRoKi FastAPI JAX solver connected' : 'Connecting to PyRoKi FastAPI...'}
          />
          <span className="text-[9px] font-mono text-slate-400 dark:text-slate-500 hidden sm:inline">
            {pyrokiAvailable ? 'PyRoKi' : 'IK'}
          </span>
          <ChevronDown
            className={`w-3.5 h-3.5 text-slate-400 transition-transform duration-200 ${
              isOpen ? 'rotate-180 text-indigo-500' : ''
            }`}
          />
        </div>
      </button>

      {/* Dropdown Menu */}
      {isOpen && (
        <div
          role="listbox"
          aria-label="Select Manipulator"
          className={`glass-panel absolute top-full mt-2 w-[340px] max-w-[90vw] rounded-2xl p-2 border backdrop-blur-2xl transition-all duration-200 divide-y z-50 animate-in fade-in zoom-in-95 ${menuTheme}`}
        >
          <div className="px-3 py-2 text-[10px] font-semibold text-slate-400 uppercase tracking-wider flex items-center justify-between">
            <span>Select Manipulator</span>
            <span className="text-[9px] font-mono text-slate-400">MuJoCo Menagerie</span>
          </div>

          <div className="py-1 flex flex-col gap-1">
            {robotList.map((robot) => {
              const isSelected = robot.id === currentRobotId;
              return (
                <button
                  key={robot.id}
                  type="button"
                  role="option"
                  aria-selected={isSelected}
                  onClick={() => {
                    if (robot.id !== currentRobotId) {
                      onSelectRobot(robot.id);
                    }
                    setIsOpen(false);
                  }}
                  className={`w-full text-left p-2.5 rounded-xl transition-all duration-150 flex items-start gap-3 cursor-pointer ${
                    isSelected
                      ? isDarkMode
                        ? 'bg-indigo-950/60 border border-indigo-500/30 text-white'
                        : 'bg-indigo-50/80 border border-indigo-200 text-slate-900'
                      : isDarkMode
                      ? 'hover:bg-white/5 text-slate-300'
                      : 'hover:bg-slate-100/70 text-slate-700'
                  }`}
                >
                  <div
                    className={`w-8 h-8 rounded-lg flex items-center justify-center shrink-0 mt-0.5 ${
                      isSelected
                        ? 'bg-indigo-600 text-white shadow-sm shadow-indigo-500/30'
                        : isDarkMode
                        ? 'bg-slate-800 text-slate-400'
                        : 'bg-slate-100 text-slate-500'
                    }`}
                  >
                    <Layers className="w-4 h-4" />
                  </div>

                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-xs font-bold truncate leading-tight">
                        {robot.name}
                      </span>
                      {isSelected && (
                        <Check className="w-3.5 h-3.5 text-indigo-500 shrink-0" />
                      )}
                    </div>

                    <p className="text-[11px] text-slate-400 dark:text-slate-400 mt-0.5 line-clamp-2 leading-relaxed">
                      {robot.description}
                    </p>

                    <div className="flex items-center gap-2 mt-2 text-[10px] font-mono text-slate-500 dark:text-slate-400">
                      <span className="flex items-center gap-1">
                        <Cpu className="w-3 h-3 text-slate-400" />
                        {robot.dof}-DOF
                      </span>
                      <span>·</span>
                      <span>Reach: {robot.specs.reach}</span>
                      <span>·</span>
                      <span className="text-indigo-500 dark:text-indigo-400 font-medium">
                        {robot.id === 'ur5e' ? 'Robotiq 2F-85' : 'Panda Gripper'}
                      </span>
                    </div>
                  </div>
                </button>
              );
            })}
          </div>

          <div className="p-2 pt-2 text-[10px] text-slate-400 dark:text-slate-500 text-center">
            {activeRobot.id === 'ur5e' ? (
              <span>Solves with PyRoKi JAX optimization engine</span>
            ) : (
              <span>Supports PyRoKi JAX and Analytical 7-DOF IK</span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
