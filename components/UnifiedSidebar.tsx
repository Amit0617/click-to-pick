/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  BoxSelect,
  Check,
  ChevronDown,
  Crosshair,
  FastForward,
  Grab,
  Info,
  Loader2,
  MousePointer2,
  RotateCcw,
  Send,
  Settings2,
  Sparkles,
  Thermometer,
  X
} from 'lucide-react';
import { useState } from 'react';
import { LogOverlay } from '../App';
import { DetectedItem, DetectType, LogEntry } from '../types';

interface UnifiedSidebarProps {
  isOpen: boolean;
  onClose: () => void;
  onSend: (prompt: string, type: DetectType, temperature: number, enableThinking: boolean, modelId: string) => void;
  onPickup: () => void;
  onClickToPick?: (selectedType?: DetectType) => void;
  isLoading: boolean;
  hasDetectedItems: boolean;
  logs: LogEntry[];
  onOpenLog: (log: LogEntry) => void;
  isDarkMode: boolean;
  isPickingUp?: boolean;
  playbackSpeed?: number;
}

/**
 * UnifiedSidebar
 * The primary interaction control panel: supporting Click to Pick (Direct Visual)
 * and Gemini Embodied Reasoning (AI), with Picked Items History.
 */
export function UnifiedSidebar({ 
  isOpen, 
  onClose, 
  onSend, 
  onPickup, 
  onClickToPick,
  isLoading, 
  hasDetectedItems, 
  logs, 
  onOpenLog, 
  isDarkMode,
  isPickingUp = false,
  playbackSpeed = 1
}: UnifiedSidebarProps) {
  // Targeting / Model Mode: default to Click to Pick as requested
  const [targetMode, setTargetMode] = useState<string>('click-to-pick');
  const [prompt, setPrompt] = useState('red cubes');
  const [type, setType] = useState<DetectType>('Points');
  const [temperature, setTemperature] = useState(0.1);
  const [enableThinking, setEnableThinking] = useState(true);
  const [showSettings, setShowSettings] = useState(false);

  if (!isOpen) return null;

  const isClickToPick = targetMode === 'click-to-pick';

  const panelBase = isDarkMode 
    ? "bg-slate-900/80 border-white/10 text-slate-100 shadow-slate-950/40" 
    : "bg-white/70 border-white/80 text-slate-800 shadow-slate-200/40";
  const headerBorder = isDarkMode ? "border-white/5 bg-white/5" : "border-slate-100 bg-white/40";
  const inputBg = isDarkMode 
    ? "bg-slate-950 border-white/5 text-slate-100 focus:ring-indigo-400/20 shadow-none" 
    : "bg-white/50 border-slate-200 text-slate-800 focus:ring-indigo-500/5 shadow-inner";
  const selectorBg = isDarkMode ? "bg-slate-800/40 border-white/5" : "bg-slate-100/50 border-slate-200/50";
  const selectorActive = isDarkMode ? "bg-slate-700 text-indigo-400" : "bg-white text-indigo-600 shadow-sm";
  const logCardBg = isDarkMode ? "bg-white/5 border-white/5 hover:bg-white/10" : "bg-white/40 border-slate-100 hover:bg-white hover:shadow-md";

  const pickedCount = logs.filter(l => l.status === 'picked').length;

  return (
    <div className={`absolute top-4 bottom-4 left-4 right-4 min-[660px]:left-auto min-[660px]:top-10 min-[660px]:right-10 min-[660px]:bottom-10 min-[660px]:w-96 glass-panel rounded-[2.5rem] flex flex-col z-40 overflow-hidden shadow-2xl transition-all border border-white/20 ${panelBase}`}>
      
      {/* Header with Mode Dropdown */}
      <div className={`p-6 border-b flex justify-between items-center ${headerBorder}`}>
        <div className="flex items-center gap-3 w-full pr-2">
          <div className="w-full">
            <div className="flex items-center justify-between mb-2">
              <h2 className="text-base font-bold leading-none">Targeting Mode</h2>
              {isClickToPick ? (
                <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                  Visual Pick
                </span>
              ) : (
                <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-purple-500/10 text-purple-400 border border-purple-500/20 flex items-center gap-1">
                  <Sparkles className="w-2.5 h-2.5" /> AI Model
                </span>
              )}
            </div>

            {/* Main Mode Dropdown */}
            <div className="relative">
              <select 
                value={targetMode}
                onChange={(e) => setTargetMode(e.target.value)}
                className={`appearance-none w-full rounded-xl border px-3 py-2.5 pr-8 text-xs font-bold transition-colors focus:outline-none focus:ring-2 focus:ring-indigo-500/20 cursor-pointer ${inputBg} ${isDarkMode ? 'border-white/10' : 'border-slate-200/80'}`}
              >
                <option value="click-to-pick">Click to Pick (Manual Visual)</option>
                <option value="gemini-robotics-er-2-preview"> Gemini Robotics ER (AI Model)</option>
                <option value="gemini-flash-latest">Gemini 2.5 Flash (AI Model)</option>
              </select>
              <ChevronDown className={`absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 pointer-events-none ${isDarkMode ? 'text-slate-400' : 'text-slate-500'}`} />
            </div>
          </div>
        </div>

        <button 
          onClick={onClose} 
          className="p-2 hover:bg-slate-200/20 rounded-full transition-colors text-slate-400 shrink-0"
          title="Close panel"
        >
          <X className="w-5 h-5" />
        </button>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 overflow-y-auto custom-scrollbar px-6 pt-3 pb-6 space-y-4">
        
        {/* Detection Type Selector: Boxes vs Points (available in both Click to Pick and Gemini) */}
        <section className="space-y-1.5">
          <div className="flex justify-between items-center px-1">
            <span className={`text-[10px] font-bold uppercase tracking-wider ${isDarkMode ? 'text-slate-400' : 'text-slate-500'}`}>
              Target Format
            </span>
            <span className={`text-[10px] ${isDarkMode ? 'text-slate-500' : 'text-slate-400'}`}>
              {type === 'Points' ? 'Center Point' : 'Bounding Box'}
            </span>
          </div>

          <div className={`p-1.5 rounded-2xl flex border ${selectorBg}`}>
            {(['Points', '2D bounding boxes'] as DetectType[]).map((t) => {
              const isActive = type === t;
              return (
                <button 
                  key={t}
                  onClick={() => setType(t)}
                  title={
                    t === '2D bounding boxes' 
                      ? 'Boxes: Define rectangular bounding region around target' 
                      : 'Points: Pinpoint exact target coordinate'
                  }
                  className={`flex-1 py-2.5 rounded-xl flex items-center justify-center gap-2 transition-all ${
                    isActive 
                      ? selectorActive
                      : 'text-slate-500 hover:text-slate-400'
                  }`}
                >
                  {t === 'Points' && <MousePointer2 className="w-3.5 h-3.5" />}
                  {t === '2D bounding boxes' && <BoxSelect className="w-3.5 h-3.5" />}
                  <span className="text-[11px] font-bold">
                    {t === '2D bounding boxes' ? 'Bounding Boxes' : 'Points'}
                  </span>
                </button>
              );
            })}
          </div>
        </section>

        {/* Dynamic Panel Content depending on Mode */}
        {isClickToPick ? (
          /* CLICK TO PICK MODE: No prompt input, no temperature, no thinking checkbox */
          <section className="space-y-3">
            <div className={`p-3.5 rounded-2xl border ${isDarkMode ? 'bg-indigo-950/20 border-indigo-500/20' : 'bg-indigo-50/60 border-indigo-100'}`}>
              <div className="flex items-start gap-2.5">
                <Crosshair className="w-4 h-4 text-indigo-500 shrink-0 mt-0.5" />
                <div className="text-xs space-y-1">
                  <p className={`font-semibold ${isDarkMode ? 'text-indigo-200' : 'text-indigo-900'}`}>
                    {type === 'Points' ? 'Point Targeting' : 'Box Targeting'}
                  </p>
                  <p className={`text-[11px] leading-relaxed ${isDarkMode ? 'text-slate-400' : 'text-slate-600'}`}>
                    {type === 'Points'
                      ? 'Takes a top-down workspace snapshot. Click directly on objects to set robotic pick coordinates.'
                      : 'Takes a top-down workspace snapshot. Draw a bounding box around objects to target their center.'}
                  </p>
                </div>
              </div>
            </div>

            {/* Click to Pick Action Trigger */}
            <button 
              onClick={() => onClickToPick?.(type)}
              disabled={isLoading || isPickingUp}
              title="Open workspace snapshot for direct targeting"
              className={`w-full py-3.5 rounded-2xl font-bold text-xs uppercase tracking-wider flex items-center justify-center gap-2 transition-all shadow-lg active:scale-[0.98] ${
                isLoading || isPickingUp
                  ? 'bg-slate-200 text-slate-400 cursor-not-allowed dark:bg-slate-800 dark:text-slate-600'
                  : 'bg-indigo-600 hover:bg-indigo-700 text-white shadow-indigo-500/20'
              }`}
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Capturing Snapshot...</span>
                </>
              ) : (
                <>
                  <Crosshair className="w-4 h-4" />
                  <span>Capture & Select Objects</span>
                </>
              )}
            </button>

            {/* If items were targeted or robot is in motion, show Pickup button */}
            {(hasDetectedItems || isPickingUp) && (
              <button 
                onClick={() => {
                  onPickup();
                  if (window.innerWidth < 660) onClose();
                }}
                disabled={isLoading}
                title={isPickingUp ? `Click to toggle speed (Current: ${playbackSpeed}x)` : "Start pickup sequence"}
                className={`w-full py-3.5 rounded-2xl font-bold text-xs uppercase tracking-wider flex items-center justify-center gap-2 transition-all shadow-xl active:scale-[0.98] ${
                  isPickingUp 
                    ? 'bg-emerald-600 hover:bg-emerald-700 text-white shadow-emerald-500/20'
                    : 'bg-slate-900 text-white hover:bg-black shadow-lg dark:bg-slate-100 dark:text-slate-900 dark:hover:bg-white'
                }`}
              >
                {isPickingUp ? (
                  <>
                    <FastForward className="w-4 h-4" /> 
                    <span>Fast Forward ({playbackSpeed}x)</span>
                  </>
                ) : (
                  <>
                    <Grab className="w-4 h-4" /> 
                    <span>Pick Up Target(s)</span>
                  </>
                )}
              </button>
            )}
          </section>
        ) : (
          /* GEMINI AI MODE: Prompt input, temperature, thinking checkbox, detect button */
          <section className="space-y-3">
            <div className="relative group">
              <textarea 
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                className={`w-full rounded-2xl px-4 py-3 pr-10 text-sm focus:outline-none transition-all resize-none h-14 border ${inputBg}`}
                placeholder="Describe targets (e.g., 'red cubes', 'stacked blocks')..."
              />
              <button
                onClick={() => setShowSettings(!showSettings)}
                className={`absolute top-1/2 right-2.5 -translate-y-1/2 p-1.5 rounded-xl transition-colors ${
                  showSettings 
                    ? (isDarkMode ? 'text-indigo-400 bg-white/10' : 'text-indigo-600 bg-slate-200') 
                    : (isDarkMode ? 'text-slate-500 hover:text-slate-300' : 'text-slate-400 hover:text-slate-600')
                }`}
                title="Toggle Model Settings"
              >
                <Settings2 className="w-4 h-4" />
              </button>
            </div>

            {/* Model Configuration Controls */}
            {showSettings && (
              <div className={`p-3 rounded-2xl border space-y-3 animate-in slide-in-from-top-2 fade-in duration-200 ${isDarkMode ? 'bg-slate-800/30 border-white/5' : 'bg-slate-50 border-slate-200/60'}`}>
                <div className="flex flex-col gap-1.5">
                  <div className="flex justify-between items-end">
                    <div className={`flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-widest ${isDarkMode ? 'text-slate-400' : 'text-slate-500'}`}>
                      <Thermometer className="w-3 h-3" />
                      <span>Temperature</span>
                    </div>
                    <span className={`text-[10px] font-mono font-bold ${isDarkMode ? 'text-slate-300' : 'text-slate-600'}`}>
                      {temperature}
                    </span>
                  </div>
                  <input 
                    type="range" 
                    min="0" 
                    max="2" 
                    step="0.1" 
                    value={temperature}
                    onChange={(e) => setTemperature(parseFloat(e.target.value))}
                    className="w-full h-1 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-indigo-600 dark:bg-slate-700"
                  />
                </div>

                <div className="flex items-center gap-2 pt-1" title="Enable reasoning chain for complex spatial positioning">
                  <input 
                    type="checkbox" 
                    id="thinking-toggle"
                    checked={enableThinking}
                    onChange={(e) => setEnableThinking(e.target.checked)}
                    className="w-3.5 h-3.5 rounded border-slate-300 text-indigo-600 focus:ring-indigo-500 cursor-pointer"
                  />
                  <label htmlFor="thinking-toggle" className={`text-[10px] font-bold uppercase tracking-widest cursor-pointer select-none ${isDarkMode ? 'text-slate-300' : 'text-slate-500'}`}>
                    Thinking
                  </label>
                </div>
              </div>
            )}
            
            <div className="flex gap-2.5">
              <button 
                onClick={() => onSend(prompt, type, temperature, enableThinking, targetMode)}
                disabled={isLoading || !prompt.trim()}
                title="Detect: Trigger Gemini analysis of current workspace"
                className={`flex-1 py-3.5 rounded-2xl font-bold text-xs uppercase tracking-wider flex items-center justify-center gap-2 transition-all ${
                  isLoading 
                    ? 'bg-slate-200 text-slate-400 cursor-not-allowed dark:bg-slate-800 dark:text-slate-600' 
                    : 'bg-slate-900 text-white hover:bg-black shadow-lg active:scale-[0.98] dark:bg-slate-100 dark:text-slate-900 dark:hover:bg-white'
                }`}
              >
                {isLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
                <span>{isLoading ? 'Detecting' : 'Detect'}</span>
              </button>

              <button 
                onClick={() => {
                  onPickup();
                  if (window.innerWidth < 660) onClose();
                }}
                disabled={(!hasDetectedItems && !isPickingUp) || isLoading}
                title={isPickingUp ? `Toggle simulation speed (Current: ${playbackSpeed}x)` : "Start pickup sequence"}
                className={`flex-1 py-3.5 rounded-2xl font-bold text-xs uppercase tracking-wider flex items-center justify-center gap-2 transition-all shadow-xl active:scale-[0.98] ${
                  (!hasDetectedItems && !isPickingUp) || isLoading 
                    ? (isDarkMode ? 'bg-slate-800 text-slate-600 cursor-not-allowed shadow-none' : 'bg-slate-100 text-slate-400 cursor-not-allowed shadow-none')
                    : (isPickingUp 
                        ? 'bg-emerald-600 hover:bg-emerald-700 text-white shadow-emerald-500/20'
                        : 'bg-indigo-600 hover:bg-indigo-700 text-white shadow-indigo-500/20'
                      )
                }`}
              >
                {isPickingUp ? (
                  <>
                    <FastForward className="w-4 h-4" /> 
                    <span>Fast ({playbackSpeed}x)</span>
                  </>
                ) : (
                  <>
                    <Grab className="w-4 h-4" /> 
                    <span>Pickup</span>
                  </>
                )}
              </button>
            </div>
          </section>
        )}

        {/* Picked Items History Section (formerly API Call History) */}
        <section className="space-y-2.5 pt-2">
          <div className="flex items-center justify-between px-1">
            <div className="flex items-center gap-2">
              <Grab className="w-4 h-4 text-indigo-500" />
              <h3 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">
                Picked Items History
              </h3>
            </div>
            {logs.length > 0 && (
              <span className={`text-[9px] font-bold px-2 py-0.5 rounded-full border ${
                isDarkMode 
                  ? 'bg-indigo-500/10 border-indigo-500/20 text-indigo-300' 
                  : 'bg-indigo-50 border-indigo-200 text-indigo-700'
              }`}>
                {pickedCount > 0 ? `${pickedCount} Picked` : `${logs.length} Logged`}
              </span>
            )}
          </div>
          
          <div className="space-y-2.5">
            {logs.length === 0 ? (
              <div className={`text-center py-8 border-2 border-dashed rounded-[2rem] text-xs italic ${
                isDarkMode ? 'border-white/5 text-slate-600' : 'border-slate-100 text-slate-400'
              }`}>
                No items targeted or picked yet
              </div>
            ) : (
              logs.map((log) => {
                const itemResult = log.result as DetectedItem[] | null;
                const errorResult = log.result as { error: string } | null;
                const errorMessage = errorResult?.error;
                const isPicked = log.status === 'picked';
                const isTargeted = log.status === 'targeted' || (!isPicked && Array.isArray(itemResult) && itemResult.length > 0);
                
                return (
                  <div 
                    key={log.id} 
                    onClick={() => onOpenLog(log)}
                    className={`group flex gap-3.5 p-3 border rounded-2xl transition-all cursor-pointer ${logCardBg}`}
                  >
                    {/* Snapshot with visual overlay */}
                    <div className={`relative w-20 rounded-xl overflow-hidden shrink-0 border self-start ${
                      isDarkMode ? 'bg-slate-950 border-white/5' : 'bg-slate-100 border-slate-200'
                    }`}>
                      <img src={log.imageSrc} className="w-full h-auto block" alt="Workspace log snapshot" />
                      <LogOverlay log={log} />
                    </div>

                    <div className="flex-1 min-w-0 py-0.5">
                      {/* Top Status & Timestamp */}
                      <div className="flex justify-between items-center mb-1">
                        <span className="text-[9px] font-bold text-slate-400">
                          {log.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                        </span>
                        
                        {isPicked ? (
                          <span className="text-[8px] font-bold px-1.5 py-0.5 rounded-full uppercase tracking-tight flex items-center gap-1 bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20">
                            <Check className="w-2.5 h-2.5" /> Picked
                          </span>
                        ) : isTargeted ? (
                          <span className="text-[8px] font-bold px-1.5 py-0.5 rounded-full uppercase tracking-tight flex items-center gap-1 bg-indigo-500/15 text-indigo-600 dark:text-indigo-400 border border-indigo-500/20">
                            <Crosshair className="w-2.5 h-2.5" /> Ready
                          </span>
                        ) : null}
                      </div>

                      {/* Prompt / Label */}
                      <p className={`text-[11px] font-semibold truncate ${
                        isDarkMode ? 'text-slate-200' : 'text-slate-700'
                      }`}>
                        {log.prompt}
                      </p>

                      {/* Details & Tags */}
                      {log.result === null ? (
                        <div className="flex items-center gap-1.5 text-indigo-400 mt-1.5 text-[10px] font-bold">
                          <Loader2 className="w-3 h-3 animate-spin" />
                          <span>Detecting...</span>
                        </div>
                      ) : errorMessage ? (
                        <div className="flex items-center justify-between mt-1.5 gap-2">
                           <div className="flex items-center gap-1.5 min-w-0">
                             <span className="text-[10px] font-bold text-red-500 truncate">Failed</span>
                             <div className="group/info relative shrink-0" title={errorMessage}>
                               <Info className="w-3 h-3 text-red-400/80 hover:text-red-500 cursor-help" />
                             </div>
                           </div>
                           <button 
                             onClick={(e) => {
                               e.stopPropagation();
                               onSend(log.prompt, log.type as DetectType, temperature, enableThinking, targetMode);
                             }}
                             className={`shrink-0 flex items-center gap-1 px-2 py-1 rounded-lg text-[9px] font-bold transition-all ${
                               isDarkMode 
                                 ? 'bg-red-500/10 text-red-400 hover:bg-red-500/20' 
                                 : 'bg-red-50 text-red-600 hover:bg-red-100'
                             }`}
                           >
                             <RotateCcw className="w-2.5 h-2.5" />
                             <span>Retry</span>
                           </button>
                        </div>
                      ) : (
                        <div className="flex items-center justify-between mt-1.5">
                          <span className={`text-[9px] font-mono ${isDarkMode ? 'text-slate-400' : 'text-slate-500'}`}>
                            {Array.isArray(itemResult) ? `${itemResult.length} item${itemResult.length > 1 ? 's' : ''}` : '0 items'}
                          </span>
                          <span className={`text-[8px] font-bold px-1.5 py-0.5 rounded uppercase tracking-tighter ${
                            log.mode === 'click-to-pick' 
                              ? (isDarkMode ? 'bg-indigo-500/10 text-indigo-400' : 'bg-indigo-50 text-indigo-600')
                              : (isDarkMode ? 'bg-white/5 text-slate-400' : 'bg-slate-100 text-slate-500')
                          }`}>
                            {log.mode === 'click-to-pick' ? 'Direct Pick' : 'Gemini'}
                          </span>
                        </div>
                      )}
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </section>

      </div>
    </div>
  );
}
