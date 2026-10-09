"use client"

import { useId, useState } from "react"
import { Download, Maximize2, TrendingUp } from "lucide-react"

const POINTS = [
  { x: 0, y: 12 },
  { x: 1, y: 24 },
  { x: 2, y: 33 },
  { x: 3, y: 30 },
  { x: 4, y: 46 },
  { x: 5, y: 58 },
  { x: 6, y: 65 },
  { x: 7, y: 79 },
]

const W = 520
const H = 220
const PAD = 28

function scaleX(x: number) {
  return PAD + (x / 7) * (W - PAD * 2)
}
function scaleY(y: number) {
  return H - PAD - (y / 90) * (H - PAD * 2)
}

export function GraphWidget({ fill }: { fill?: boolean }) {
  const gradientId = useId()
  const [hover, setHover] = useState<number | null>(null)

  const linePath = POINTS.map(
    (p, i) => `${i === 0 ? "M" : "L"} ${scaleX(p.x)} ${scaleY(p.y)}`,
  ).join(" ")
  const areaPath = `${linePath} L ${scaleX(7)} ${H - PAD} L ${scaleX(0)} ${H - PAD} Z`

  return (
    <div
      className={
        fill
          ? "flex h-full flex-col rounded-2xl border border-border bg-card/80 p-4 shadow-sm backdrop-blur-sm"
          : "rounded-2xl border border-border bg-card p-4 shadow-sm"
      }
    >
      <div className="mb-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="bg-brand-gradient grid size-8 place-items-center rounded-lg text-white">
            <TrendingUp className="size-4" aria-hidden="true" />
          </span>
          <div>
            <h3 className="font-display text-sm font-semibold">
              Torque vs. Angular Acceleration
            </h3>
            <p className="text-xs text-muted-foreground">
              {"\u03C4 = I\u03B1 · linear fit"}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-1">
          <IconBtn label="Save graph">
            <Download className="size-4" aria-hidden="true" />
          </IconBtn>
          <IconBtn label="Expand graph">
            <Maximize2 className="size-4" aria-hidden="true" />
          </IconBtn>
        </div>
      </div>

      <svg
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio={fill ? "xMidYMid meet" : undefined}
        className={fill ? "min-h-0 w-full flex-1" : "h-auto w-full"}
        role="img"
        aria-label="Line chart showing torque increasing with angular acceleration"
      >
        <defs>
          <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--brand-red)" stopOpacity="0.35" />
            <stop offset="100%" stopColor="var(--brand-purple)" stopOpacity="0.02" />
          </linearGradient>
          <linearGradient id={`${gradientId}-line`} x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="var(--brand-red)" />
            <stop offset="100%" stopColor="var(--brand-purple)" />
          </linearGradient>
        </defs>

        {/* gridlines */}
        {[0, 1, 2, 3].map((i) => {
          const y = PAD + (i / 3) * (H - PAD * 2)
          return (
            <line
              key={i}
              x1={PAD}
              x2={W - PAD}
              y1={y}
              y2={y}
              stroke="var(--border)"
              strokeWidth="1"
              strokeDasharray="3 5"
            />
          )
        })}

        {/* axes */}
        <line x1={PAD} y1={H - PAD} x2={W - PAD} y2={H - PAD} stroke="var(--border)" strokeWidth="1.5" />
        <line x1={PAD} y1={PAD} x2={PAD} y2={H - PAD} stroke="var(--border)" strokeWidth="1.5" />

        <path d={areaPath} fill={`url(#${gradientId})`} />
        <path
          d={linePath}
          fill="none"
          stroke={`url(#${gradientId}-line)`}
          strokeWidth="2.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />

        {POINTS.map((p, i) => (
          <g key={i}>
            <circle
              cx={scaleX(p.x)}
              cy={scaleY(p.y)}
              r={hover === i ? 6 : 4}
              fill="var(--card)"
              stroke="var(--brand)"
              strokeWidth="2.5"
              className="transition-all"
              onMouseEnter={() => setHover(i)}
              onMouseLeave={() => setHover(null)}
            />
            {hover === i && (
              <text
                x={scaleX(p.x)}
                y={scaleY(p.y) - 12}
                textAnchor="middle"
                className="fill-foreground text-[10px] font-semibold"
              >
                {p.y} N·m
              </text>
            )}
          </g>
        ))}
      </svg>
    </div>
  )
}

function IconBtn({
  children,
  label,
}: {
  children: React.ReactNode
  label: string
}) {
  return (
    <button
      type="button"
      aria-label={label}
      className="grid size-8 place-items-center rounded-lg text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
    >
      {children}
    </button>
  )
}
