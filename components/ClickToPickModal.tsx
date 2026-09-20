// React 19's JSX runtime types are not available in the current project type setup.
// Keep this component buildable until the project's React type dependencies are fixed.
// @ts-nocheck

/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { Check, Crosshair, Grab, MousePointer2, RotateCcw, X } from 'lucide-react';
import React, { useEffect, useRef, useState } from 'react';

interface Point {
  x: number;
  y: number;
}

interface ClickToPickModalProps {
  imageSrc: string;
  isDarkMode: boolean;
  onConfirm: (points: Point[], autoPickup?: boolean) => void;
  onCancel: () => void;
}

/**
 * ClickToPickModal
 * Displays the top-down snapshot of the workspace and captures the (x, y) coordinates of user clicks.
 * Coordinates are normalized to 0-1000 matching Gemini Embodied Reasoning points convention.
 */
export function ClickToPickModal({
  imageSrc,
  isDarkMode,
  onConfirm,
  onCancel,
}: ClickToPickModalProps) {
  const imageRef = useRef<HTMLImageElement>(null);
  const [points, setPoints] = useState<Point[]>([]);
  const [hoverCoords, setHoverCoords] = useState<Point | null>(null);
  const [isMultiMode, setIsMultiMode] = useState(false);
  const [lastClickedPoint, setLastClickedPoint] = useState<Point | null>(null);

  // Keyboard navigation: Escape cancels, Enter confirms
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onCancel();
      } else if (e.key === 'Enter' && points.length > 0) {
        onConfirm(points, false);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [points, onCancel, onConfirm]);

  const handleImageMouseMove = (e: React.MouseEvent<HTMLImageElement>) => {
    if (!imageRef.current) return;
    const rect = imageRef.current.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;
    const normX = Math.round(Math.max(0, Math.min(1000, (clickX / rect.width) * 1000)));
    const normY = Math.round(Math.max(0, Math.min(1000, (clickY / rect.height) * 1000)));
    setHoverCoords({ x: normX, y: normY });
  };

  const handleImageMouseLeave = () => {
    setHoverCoords(null);
  };

  const handleImageClick = (e: React.MouseEvent<HTMLImageElement>) => {
    if (!imageRef.current) return;
    const rect = imageRef.current.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;

    // Normalizing coordinates to 0-1000 scale
    const normX = Math.round(Math.max(0, Math.min(1000, (clickX / rect.width) * 1000)));
    const normY = Math.round(Math.max(0, Math.min(1000, (clickY / rect.height) * 1000)));

    const newPoint: Point = { x: normX, y: normY };
    setLastClickedPoint(newPoint);

    if (isMultiMode) {
      // In multi-mode, check if clicked very close to an existing point to remove it
      const existingIdx = points.findIndex(
        (p) => Math.hypot(p.x - normX, p.y - normY) < 30
      );
      if (existingIdx !== -1) {
        setPoints((prev) => prev.filter((_, idx) => idx !== existingIdx));
      } else {
        setPoints((prev) => [...prev, newPoint]);
      }
    } else {
      // In single mode, replace with the clicked target
      setPoints([newPoint]);
    }
  };

  // Double click confirms immediately
  const handleImageDoubleClick = (e: React.MouseEvent<HTMLImageElement>) => {
    if (!imageRef.current) return;
    const rect = imageRef.current.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;
    const normX = Math.round(Math.max(0, Math.min(1000, (clickX / rect.width) * 1000)));
    const normY = Math.round(Math.max(0, Math.min(1000, (clickY / rect.height) * 1000)));
    onConfirm([{ x: normX, y: normY }], false);
  };

  const panelBase = isDarkMode
    ? 'bg-slate-900/90 border-white/10 text-slate-100'
    : 'bg-white/90 border-white/80 text-slate-800';

  return (
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center p-4 min-[660px]:p-8 bg-slate-950/40 backdrop-blur-xl animate-in fade-in duration-200"
      onClick={onCancel}
    >
      <div
        className={`glass-panel w-full max-w-3xl rounded-[2.5rem] shadow-2xl flex flex-col overflow-hidden border transition-all ${panelBase}`}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div
          className={`p-6 border-b flex justify-between items-center shrink-0 ${
            isDarkMode ? 'border-white/5 bg-white/5' : 'border-slate-100 bg-white/40'
          }`}
        >
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-2xl bg-indigo-600 flex items-center justify-center shadow-lg shadow-indigo-500/20 text-white">
              <Crosshair className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-lg font-bold leading-tight">Click to Pick</h3>
              <p
                className={`text-xs font-medium ${
                  isDarkMode ? 'text-slate-400' : 'text-slate-500'
                }`}
              >
                Click any object in the top-down snapshot to target it
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {/* Multi-target toggle */}
            <button
              onClick={() => {
                setIsMultiMode(!isMultiMode);
                if (!isMultiMode && points.length > 1) {
                  setPoints(points.slice(0, 1));
                }
              }}
              className={`px-3 py-1.5 rounded-xl text-xs font-bold transition-all border flex items-center gap-1.5 ${
                isMultiMode
                  ? 'bg-indigo-600 text-white border-indigo-600 shadow-sm'
                  : isDarkMode
                  ? 'bg-slate-800 text-slate-400 border-white/10 hover:text-slate-200'
                  : 'bg-slate-100 text-slate-600 border-slate-200 hover:bg-slate-200'
              }`}
              title="Toggle selecting multiple objects in one go"
            >
              <MousePointer2 className="w-3.5 h-3.5" />
              <span>{isMultiMode ? 'Multi-Target Active' : 'Single Target'}</span>
            </button>

            <button
              onClick={onCancel}
              className={`w-9 h-9 flex items-center justify-center rounded-full shadow-sm border transition-colors ${
                isDarkMode
                  ? 'bg-slate-800 border-white/10 text-slate-400 hover:text-slate-200'
                  : 'bg-white border-slate-200 text-slate-400 hover:text-slate-600'
              }`}
              title="Close and cancel (Esc)"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Snapshot Interactive Display */}
        <div
          className={`relative p-6 flex flex-col items-center justify-center overflow-hidden select-none ${
            isDarkMode ? 'bg-slate-950/60' : 'bg-slate-100/60'
          }`}
        >
          <div className="relative inline-block rounded-2xl overflow-hidden shadow-2xl border-2 border-indigo-500/30 group">
            <img
              ref={imageRef}
              src={imageSrc}
              alt="Top-down workspace snapshot"
              onMouseMove={handleImageMouseMove}
              onMouseLeave={handleImageMouseLeave}
              onClick={handleImageClick}
              onDoubleClick={handleImageDoubleClick}
              className="max-h-[55vh] w-auto h-auto object-contain block cursor-crosshair"
              draggable={false}
            />

            {/* SVG overlay for markers & target points */}
            <svg
              viewBox="0 0 1000 1000"
              preserveAspectRatio="none"
              className="absolute inset-0 pointer-events-none w-full h-full z-10"
            >
              {points.map((pt, idx) => (
                <g key={idx}>
                  {/* Outer pulsating ring */}
                  <circle
                    cx={pt.x}
                    cy={pt.y}
                    r="24"
                    fill="rgba(79, 70, 229, 0.25)"
                    stroke="#4f46e5"
                    strokeWidth="2"
                    vectorEffect="non-scaling-stroke"
                    className="animate-ping"
                  />
                  {/* Inner targeting reticle */}
                  <circle
                    cx={pt.x}
                    cy={pt.y}
                    r="12"
                    fill="#4f46e5"
                    stroke="white"
                    strokeWidth="2.5"
                    vectorEffect="non-scaling-stroke"
                  />
                  {/* Center precision dot */}
                  <circle
                    cx={pt.x}
                    cy={pt.y}
                    r="3"
                    fill="white"
                    vectorEffect="non-scaling-stroke"
                  />
                  {/* Crosshair lines */}
                  <line
                    x1={pt.x - 18}
                    y1={pt.y}
                    x2={pt.x + 18}
                    y2={pt.y}
                    stroke="#4f46e5"
                    strokeWidth="1.5"
                    vectorEffect="non-scaling-stroke"
                  />
                  <line
                    x1={pt.x}
                    y1={pt.y - 18}
                    x2={pt.x}
                    y2={pt.y + 18}
                    stroke="#4f46e5"
                    strokeWidth="1.5"
                    vectorEffect="non-scaling-stroke"
                  />
                  {/* Number Badge if multiple */}
                  {points.length > 1 && (
                    <text
                      x={pt.x + 16}
                      y={pt.y - 12}
                      fill="white"
                      stroke="#4f46e5"
                      strokeWidth="3"
                      paintOrder="stroke"
                      fontSize="24"
                      fontWeight="bold"
                      fontFamily="sans-serif"
                    >
                      {idx + 1}
                    </text>
                  )}
                </g>
              ))}
            </svg>

            {/* Live Hover Coordinates Tooltip Badge */}
            {hoverCoords && (
              <div
                className={`absolute bottom-3 left-3 px-2.5 py-1 rounded-lg font-mono text-[11px] font-bold pointer-events-none backdrop-blur-md shadow-md border ${
                  isDarkMode
                    ? 'bg-slate-900/80 text-indigo-300 border-white/10'
                    : 'bg-white/90 text-indigo-700 border-slate-200'
                }`}
              >
                X: {hoverCoords.x} &nbsp; Y: {hoverCoords.y}
              </div>
            )}
          </div>

          {/* Quick Helper Tip */}
          <div className="mt-3 flex items-center justify-between w-full max-w-md px-2 text-[11px]">
            <span
              className={`font-medium ${
                isDarkMode ? 'text-slate-400' : 'text-slate-500'
              }`}
            >
              {points.length === 0
                ? 'Click on any cube or object in the workspace to place target'
                : `${points.length} target${points.length > 1 ? 's' : ''} placed (Double click image or press Enter to confirm)`}
            </span>

            {points.length > 0 && (
              <button
                onClick={() => setPoints([])}
                className="text-indigo-500 hover:text-indigo-600 font-bold flex items-center gap-1"
              >
                <RotateCcw className="w-3 h-3" />
                <span>Clear</span>
              </button>
            )}
          </div>
        </div>

        {/* Action Controls Footer */}
        <div
          className={`p-6 border-t flex flex-wrap gap-3 items-center justify-between shrink-0 ${
            isDarkMode ? 'border-white/5 bg-white/5' : 'border-slate-100 bg-white/40'
          }`}
        >
          <div className="flex items-center gap-2">
            {lastClickedPoint ? (
              <div
                className={`text-xs px-3 py-1.5 rounded-xl font-mono border ${
                  isDarkMode
                    ? 'bg-slate-800/80 border-white/10 text-slate-300'
                    : 'bg-slate-100 border-slate-200 text-slate-700'
                }`}
              >
                Target: X: <strong>{lastClickedPoint.x}</strong>, Y:{' '}
                <strong>{lastClickedPoint.y}</strong>
              </div>
            ) : (
              <div
                className={`text-xs italic ${
                  isDarkMode ? 'text-slate-500' : 'text-slate-400'
                }`}
              >
                Awaiting your click on the snapshot...
              </div>
            )}
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={onCancel}
              className={`px-5 py-2.5 rounded-2xl font-bold text-xs transition-colors border ${
                isDarkMode
                  ? 'bg-slate-800 text-slate-300 border-white/10 hover:bg-slate-700'
                  : 'bg-slate-100 text-slate-600 border-slate-200 hover:bg-slate-200'
              }`}
            >
              Cancel
            </button>

            <button
              onClick={() => onConfirm(points, false)}
              disabled={points.length === 0}
              className={`px-5 py-2.5 rounded-2xl font-bold text-xs flex items-center gap-2 transition-all shadow-lg active:scale-95 ${
                points.length === 0
                  ? 'bg-slate-200 text-slate-400 cursor-not-allowed dark:bg-slate-800 dark:text-slate-600 shadow-none'
                  : 'bg-indigo-600 hover:bg-indigo-700 text-white shadow-indigo-500/25'
              }`}
            >
              <Check className="w-4 h-4" />
              <span>Confirm Target{points.length > 1 ? `s (${points.length})` : ''}</span>
            </button>

            <button
              onClick={() => onConfirm(points, true)}
              disabled={points.length === 0}
              className={`px-5 py-2.5 rounded-2xl font-bold text-xs flex items-center gap-2 transition-all shadow-lg active:scale-95 ${
                points.length === 0
                  ? 'bg-slate-200 text-slate-400 cursor-not-allowed dark:bg-slate-800 dark:text-slate-600 shadow-none'
                  : 'bg-emerald-600 hover:bg-emerald-700 text-white shadow-emerald-500/25'
              }`}
              title="Confirm selection and immediately start robot pickup sequence"
            >
              <Grab className="w-4 h-4" />
              <span>Confirm & Pick Up</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
