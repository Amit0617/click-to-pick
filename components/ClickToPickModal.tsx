// React 19's JSX runtime types are not available in the current project type setup.
// Keep this component buildable until the project's React type dependencies are fixed.
// @ts-nocheck

/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { BoxSelect, Check, Crosshair, Grab, MousePointer2, RotateCcw, X } from 'lucide-react';
import React, { useEffect, useRef, useState } from 'react';
import { DetectedItem, DetectType } from '../types';

interface ClickToPickModalProps {
  imageSrc: string;
  initialType?: DetectType;
  isDarkMode: boolean;
  onConfirm: (items: DetectedItem[], autoPickup?: boolean) => void;
  onCancel: () => void;
}

interface Point2D {
  x: number;
  y: number;
}

interface Box2D {
  ymin: number;
  xmin: number;
  ymax: number;
  xmax: number;
}

/**
 * ClickToPickModal
 * Displays the top-down snapshot of the workspace and allows the user to target objects
 * either by clicking points or by drawing 2D bounding boxes.
 * All coordinates are normalized to 0-1000 matching Gemini Embodied Reasoning conventions.
 */
export function ClickToPickModal({
  imageSrc,
  initialType = 'Points',
  isDarkMode,
  onConfirm,
  onCancel,
}: ClickToPickModalProps) {
  const imageRef = useRef<HTMLImageElement>(null);
  const [activeType, setActiveType] = useState<DetectType>(initialType);
  const [isMultiMode, setIsMultiMode] = useState(false);

  // Points state
  const [points, setPoints] = useState<Point2D[]>([]);

  // Boxes state
  const [boxes, setBoxes] = useState<Box2D[]>([]);
  const [dragStart, setDragStart] = useState<Point2D | null>(null);
  const [dragCurrent, setDragCurrent] = useState<Point2D | null>(null);

  // Hover tracker
  const [hoverCoords, setHoverCoords] = useState<Point2D | null>(null);

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onCancel();
      } else if (e.key === 'Enter') {
        handleConfirmAction(false);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  });

  const getNormCoords = (e: React.MouseEvent<HTMLImageElement | SVGSVGElement>): Point2D | null => {
    if (!imageRef.current) return null;
    const rect = imageRef.current.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;
    const normX = Math.round(Math.max(0, Math.min(1000, (clickX / rect.width) * 1000)));
    const normY = Math.round(Math.max(0, Math.min(1000, (clickY / rect.height) * 1000)));
    return { x: normX, y: normY };
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLImageElement | SVGSVGElement>) => {
    const coords = getNormCoords(e);
    if (!coords) return;
    setHoverCoords(coords);

    if (dragStart && activeType === '2D bounding boxes') {
      setDragCurrent(coords);
    }
  };

  const handleMouseLeave = () => {
    setHoverCoords(null);
    if (dragStart) {
      setDragStart(null);
      setDragCurrent(null);
    }
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLImageElement | SVGSVGElement>) => {
    // Only primary mouse button
    if (e.button !== 0) return;
    const coords = getNormCoords(e);
    if (!coords) return;

    if (activeType === '2D bounding boxes') {
      setDragStart(coords);
      setDragCurrent(coords);
    }
  };

  const handleMouseUp = (e: React.MouseEvent<HTMLImageElement | SVGSVGElement>) => {
    if (e.button !== 0) return;
    const coords = getNormCoords(e);

    if (activeType === '2D bounding boxes' && dragStart) {
      const endCoords = coords || dragCurrent || dragStart;
      const xmin = Math.min(dragStart.x, endCoords.x);
      const xmax = Math.max(dragStart.x, endCoords.x);
      const ymin = Math.min(dragStart.y, endCoords.y);
      const ymax = Math.max(dragStart.y, endCoords.y);

      // Require a minimum size to avoid accidental click-drag of 0px
      if (xmax - xmin >= 15 && ymax - ymin >= 15) {
        const newBox: Box2D = { ymin, xmin, ymax, xmax };
        if (isMultiMode) {
          setBoxes((prev) => [...prev, newBox]);
        } else {
          setBoxes([newBox]);
        }
      }
      setDragStart(null);
      setDragCurrent(null);
    }
  };

  const handleClick = (e: React.MouseEvent<HTMLImageElement | SVGSVGElement>) => {
    if (activeType !== 'Points') return;
    const coords = getNormCoords(e);
    if (!coords) return;

    if (isMultiMode) {
      const existingIdx = points.findIndex(
        (p) => Math.hypot(p.x - coords.x, p.y - coords.y) < 30
      );
      if (existingIdx !== -1) {
        setPoints((prev) => prev.filter((_, idx) => idx !== existingIdx));
      } else {
        setPoints((prev) => [...prev, coords]);
      }
    } else {
      setPoints([coords]);
    }
  };

  const hasSelection = activeType === 'Points' ? points.length > 0 : boxes.length > 0;

  const handleConfirmAction = (autoPickup = false) => {
    if (!hasSelection) return;

    const items: DetectedItem[] = [];

    if (activeType === 'Points') {
      points.forEach((pt, idx) => {
        items.push({
          point: [pt.y, pt.x],
          label: points.length > 1 ? `Target ${idx + 1}` : 'Target (Point)',
        });
      });
    } else {
      boxes.forEach((b, idx) => {
        items.push({
          box_2d: [b.ymin, b.xmin, b.ymax, b.xmax],
          label: boxes.length > 1 ? `Target ${idx + 1}` : 'Target (Box)',
        });
      });
    }

    onConfirm(items, autoPickup);
  };

  const clearSelection = () => {
    setPoints([]);
    setBoxes([]);
    setDragStart(null);
    setDragCurrent(null);
  };

  // Preview box while dragging
  const liveBox: Box2D | null =
    dragStart && dragCurrent && activeType === '2D bounding boxes'
      ? {
          xmin: Math.min(dragStart.x, dragCurrent.x),
          xmax: Math.max(dragStart.x, dragCurrent.x),
          ymin: Math.min(dragStart.y, dragCurrent.y),
          ymax: Math.max(dragStart.y, dragCurrent.y),
        }
      : null;

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
                {activeType === 'Points'
                  ? 'Click anywhere on an object to pinpoint its coordinates'
                  : 'Click and drag to draw a bounding box around an object'}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2.5">
            {/* Mode Toggle inside Modal */}
            <div
              className={`p-1 rounded-xl flex border ${
                isDarkMode ? 'bg-slate-800/80 border-white/10' : 'bg-slate-100 border-slate-200'
              }`}
            >
              <button
                onClick={() => {
                  setActiveType('Points');
                  clearSelection();
                }}
                className={`px-3 py-1.5 rounded-lg text-xs font-bold flex items-center gap-1.5 transition-all ${
                  activeType === 'Points'
                    ? 'bg-indigo-600 text-white shadow-sm'
                    : isDarkMode
                    ? 'text-slate-400 hover:text-slate-200'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
                title="Point Mode: Click on object"
              >
                <MousePointer2 className="w-3.5 h-3.5" />
                <span>Points</span>
              </button>
              <button
                onClick={() => {
                  setActiveType('2D bounding boxes');
                  clearSelection();
                }}
                className={`px-3 py-1.5 rounded-lg text-xs font-bold flex items-center gap-1.5 transition-all ${
                  activeType === '2D bounding boxes'
                    ? 'bg-indigo-600 text-white shadow-sm'
                    : isDarkMode
                    ? 'text-slate-400 hover:text-slate-200'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
                title="Box Mode: Draw rectangular bounding box"
              >
                <BoxSelect className="w-3.5 h-3.5" />
                <span>Boxes</span>
              </button>
            </div>

            {/* Multi-target toggle */}
            <button
              onClick={() => {
                setIsMultiMode(!isMultiMode);
                if (!isMultiMode) {
                  if (points.length > 1) setPoints(points.slice(0, 1));
                  if (boxes.length > 1) setBoxes(boxes.slice(0, 1));
                }
              }}
              className={`px-3 py-1.5 rounded-xl text-xs font-bold transition-all border flex items-center gap-1.5 ${
                isMultiMode
                  ? 'bg-indigo-600 text-white border-indigo-600 shadow-sm'
                  : isDarkMode
                  ? 'bg-slate-800 text-slate-400 border-white/10 hover:text-slate-200'
                  : 'bg-slate-100 text-slate-600 border-slate-200 hover:bg-slate-200'
              }`}
              title="Toggle selecting multiple objects"
            >
              <span>{isMultiMode ? 'Multi-Target' : 'Single Target'}</span>
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
              className="max-h-[55vh] w-auto h-auto object-contain block cursor-crosshair"
              draggable={false}
            />

            {/* Interactive SVG Overlay capturing mouse events */}
            <svg
              viewBox="0 0 1000 1000"
              preserveAspectRatio="none"
              className="absolute inset-0 w-full h-full z-10 cursor-crosshair"
              onMouseMove={handleMouseMove}
              onMouseLeave={handleMouseLeave}
              onMouseDown={handleMouseDown}
              onMouseUp={handleMouseUp}
              onClick={handleClick}
            >
              {/* Render existing Points */}
              {points.map((pt, idx) => (
                <g key={`pt-${idx}`}>
                  {/* Outer pulsating ring */}
                  <circle
                    cx={pt.x}
                    cy={pt.y}
                    r="12"
                    fill="rgba(79, 70, 229, 0.2)"
                    stroke="#4f46e5"
                    strokeWidth="2"
                    vectorEffect="non-scaling-stroke"
                  >
                    <animate
                      attributeName="r"
                      values="12;28"
                      dur="1.5s"
                      repeatCount="indefinite"
                    />
                    <animate
                      attributeName="opacity"
                      values="1;0"
                      dur="1.5s"
                      repeatCount="indefinite"
                    />
                  </circle>
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
                    x1={pt.x - 16}
                    y1={pt.y}
                    x2={pt.x + 16}
                    y2={pt.y}
                    stroke="#4f46e5"
                    strokeWidth="1.5"
                    vectorEffect="non-scaling-stroke"
                  />
                  <line
                    x1={pt.x}
                    y1={pt.y - 16}
                    x2={pt.x}
                    y2={pt.y + 16}
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

              {/* Render existing Bounding Boxes */}
              {boxes.map((b, idx) => {
                const centerX = (b.xmin + b.xmax) / 2;
                const centerY = (b.ymin + b.ymax) / 2;
                return (
                  <g key={`box-${idx}`}>
                    <rect
                      x={b.xmin}
                      y={b.ymin}
                      width={b.xmax - b.xmin}
                      height={b.ymax - b.ymin}
                      fill="rgba(79, 70, 229, 0.2)"
                      stroke="#4f46e5"
                      strokeWidth="2.5"
                      vectorEffect="non-scaling-stroke"
                      rx="6"
                    />
                    {/* Center grasp point indicator */}
                    <circle
                      cx={centerX}
                      cy={centerY}
                      r="6"
                      fill="rgba(79, 70, 229, 0.25)"
                      stroke="#4f46e5"
                      strokeWidth="1.5"
                      vectorEffect="non-scaling-stroke"
                    >
                      <animate
                        attributeName="r"
                        values="6;18"
                        dur="1.5s"
                        repeatCount="indefinite"
                      />
                      <animate
                        attributeName="opacity"
                        values="1;0"
                        dur="1.5s"
                        repeatCount="indefinite"
                      />
                    </circle>
                    <circle
                      cx={centerX}
                      cy={centerY}
                      r="6"
                      fill="#4f46e5"
                      stroke="white"
                      strokeWidth="2"
                      vectorEffect="non-scaling-stroke"
                    />
                    {/* Label Badge */}
                    <rect
                      x={b.xmin}
                      y={Math.max(0, b.ymin - 28)}
                      width="70"
                      height="24"
                      fill="#4f46e5"
                      rx="4"
                    />
                    <text
                      x={b.xmin + 6}
                      y={Math.max(0, b.ymin - 28) + 16}
                      fill="white"
                      fontSize="14"
                      fontWeight="bold"
                      fontFamily="sans-serif"
                    >
                      Target {boxes.length > 1 ? idx + 1 : ''}
                    </text>
                  </g>
                );
              })}

              {/* Render live dragged box */}
              {liveBox && (
                <rect
                  x={liveBox.xmin}
                  y={liveBox.ymin}
                  width={liveBox.xmax - liveBox.xmin}
                  height={liveBox.ymax - liveBox.ymin}
                  fill="rgba(99, 102, 241, 0.3)"
                  stroke="#6366f1"
                  strokeWidth="2"
                  strokeDasharray="4 3"
                  vectorEffect="non-scaling-stroke"
                />
              )}
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
              {!hasSelection
                ? activeType === 'Points'
                  ? 'Click directly on any object to target it'
                  : 'Drag across an object to draw a bounding box'
                : `${
                    activeType === 'Points' ? points.length : boxes.length
                  } target(s) placed (Press Enter to confirm)`}
            </span>

            {hasSelection && (
              <button
                onClick={clearSelection}
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
            {hasSelection ? (
              <div
                className={`text-xs px-3 py-1.5 rounded-xl font-mono border ${
                  isDarkMode
                    ? 'bg-slate-800/80 border-white/10 text-slate-300'
                    : 'bg-slate-100 border-slate-200 text-slate-700'
                }`}
              >
                {activeType === 'Points'
                  ? `Points: ${points.map((p) => `(${p.x},${p.y})`).join(', ')}`
                  : `Boxes: ${boxes
                      .map((b) => `[${b.ymin},${b.xmin},${b.ymax},${b.xmax}]`)
                      .join(', ')}`}
              </div>
            ) : (
              <div
                className={`text-xs italic ${
                  isDarkMode ? 'text-slate-500' : 'text-slate-400'
                }`}
              >
                {activeType === 'Points'
                  ? 'Awaiting point click on snapshot...'
                  : 'Awaiting box drag on snapshot...'}
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
              onClick={() => handleConfirmAction(false)}
              disabled={!hasSelection}
              className={`px-5 py-2.5 rounded-2xl font-bold text-xs flex items-center gap-2 transition-all shadow-lg active:scale-95 ${
                !hasSelection
                  ? 'bg-slate-200 text-slate-400 cursor-not-allowed dark:bg-slate-800 dark:text-slate-600 shadow-none'
                  : 'bg-indigo-600 hover:bg-indigo-700 text-white shadow-indigo-500/25'
              }`}
            >
              <Check className="w-4 h-4" />
              <span>Confirm Target{activeType === 'Points' && points.length > 1 ? `s (${points.length})` : ''}</span>
            </button>

            <button
              onClick={() => handleConfirmAction(true)}
              disabled={!hasSelection}
              className={`px-5 py-2.5 rounded-2xl font-bold text-xs flex items-center gap-2 transition-all shadow-lg active:scale-95 ${
                !hasSelection
                  ? 'bg-slate-200 text-slate-400 cursor-not-allowed dark:bg-slate-800 dark:text-slate-600 shadow-none'
                  : 'bg-emerald-600 hover:bg-emerald-700 text-white shadow-emerald-500/25'
              }`}
              title="Confirm targets and immediately initiate robot pickup sequence"
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
