"use client"

import { useId } from "react"

const W = 520
const H = 220
const PAD = 28

const PIE_COLORS = [
  "var(--brand-red)",
  "var(--brand-purple)",
  "var(--brand)",
  "color-mix(in oklab, var(--brand-red) 60%, var(--brand-purple))",
  "color-mix(in oklab, var(--brand) 60%, var(--brand-purple))",
]

function scaleX(i: number, count: number) {
  if (count <= 1) return W / 2
  return PAD + (i / (count - 1)) * (W - PAD * 2)
}
function scaleY(v: number, maxValue: number) {
  return H - PAD - (Math.abs(v) / maxValue) * (H - PAD * 2)
}

function EmptyState() {
  return (
    <div className="grid h-full place-items-center text-sm text-muted-foreground">
      No chart data returned for this result.
    </div>
  )
}

function LineChartSvg({
  labels,
  values,
  maxValue,
  gradientId,
}: {
  labels: string[]
  values: number[]
  maxValue: number
  gradientId: string
}) {
  if (values.length === 0) return <EmptyState />

  const linePath = values
    .map((v, i) => `${i === 0 ? "M" : "L"} ${scaleX(i, values.length)} ${scaleY(v, maxValue)}`)
    .join(" ")
  const areaPath = `${linePath} L ${scaleX(values.length - 1, values.length)} ${H - PAD} L ${scaleX(0, values.length)} ${H - PAD} Z`

  return (
    <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet" className="h-full w-full" role="img">
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
      {values.map((v, i) => (
        <g key={i}>
          <circle
            cx={scaleX(i, values.length)}
            cy={scaleY(v, maxValue)}
            r={4}
            fill="var(--card)"
            stroke="var(--brand)"
            strokeWidth="2.5"
          />
          <text
            x={scaleX(i, values.length)}
            y={H - PAD + 16}
            textAnchor="middle"
            className="fill-muted-foreground text-[9px]"
          >
            {labels[i] ?? ""}
          </text>
        </g>
      ))}
    </svg>
  )
}

function BarChartSvg({
  labels,
  values,
  maxValue,
  gradientId,
}: {
  labels: string[]
  values: number[]
  maxValue: number
  gradientId: string
}) {
  if (values.length === 0) return <EmptyState />

  const slot = (W - PAD * 2) / values.length
  const barWidth = Math.min(48, slot * 0.6)

  return (
    <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet" className="h-full w-full" role="img">
      <defs>
        <linearGradient id={`${gradientId}-bar`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="var(--brand-red)" />
          <stop offset="100%" stopColor="var(--brand-purple)" />
        </linearGradient>
      </defs>
      <line x1={PAD} y1={H - PAD} x2={W - PAD} y2={H - PAD} stroke="var(--border)" strokeWidth="1.5" />
      {values.map((v, i) => {
        const cx = PAD + slot * i + slot / 2
        const barHeight = (Math.abs(v) / maxValue) * (H - PAD * 2)
        return (
          <g key={i}>
            <rect
              x={cx - barWidth / 2}
              y={H - PAD - barHeight}
              width={barWidth}
              height={barHeight}
              rx={4}
              fill={`url(#${gradientId}-bar)`}
            />
            <text x={cx} y={H - PAD + 16} textAnchor="middle" className="fill-muted-foreground text-[9px]">
              {labels[i] ?? ""}
            </text>
            <text
              x={cx}
              y={H - PAD - barHeight - 6}
              textAnchor="middle"
              className="fill-foreground text-[9px] font-semibold"
            >
              {v}
            </text>
          </g>
        )
      })}
    </svg>
  )
}

function PieChartSvg({
  labels,
  values,
}: {
  labels: string[]
  values: number[]
  gradientId: string
}) {
  if (values.length === 0) return <EmptyState />

  const total = values.reduce((sum, v) => sum + Math.abs(v), 0) || 1
  const cx = W / 2
  const cy = H / 2
  const r = Math.min(W, H) / 2 - PAD

  let cumulativeAngle = -Math.PI / 2
  const slices = values.map((v, i) => {
    const fraction = Math.abs(v) / total
    const startAngle = cumulativeAngle
    const endAngle = cumulativeAngle + fraction * Math.PI * 2
    cumulativeAngle = endAngle

    const x1 = cx + r * Math.cos(startAngle)
    const y1 = cy + r * Math.sin(startAngle)
    const x2 = cx + r * Math.cos(endAngle)
    const y2 = cy + r * Math.sin(endAngle)
    const largeArc = fraction > 0.5 ? 1 : 0

    return {
      path: `M ${cx} ${cy} L ${x1} ${y1} A ${r} ${r} 0 ${largeArc} 1 ${x2} ${y2} Z`,
      color: PIE_COLORS[i % PIE_COLORS.length],
      label: labels[i] ?? `#${i + 1}`,
      value: v,
    }
  })

  return (
    <div className="flex h-full items-center gap-6">
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet" className="h-full flex-1" role="img">
        {slices.map((slice, i) => (
          <path key={i} d={slice.path} fill={slice.color} stroke="var(--card)" strokeWidth="2" />
        ))}
      </svg>
      <ul className="flex shrink-0 flex-col gap-1.5">
        {slices.map((slice, i) => (
          <li key={i} className="flex items-center gap-2 text-xs text-muted-foreground">
            <span className="size-2.5 shrink-0 rounded-full" style={{ backgroundColor: slice.color }} />
            <span className="font-medium text-foreground">{slice.label}</span>
            <span>{slice.value}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}

export function BoardChartPlot({
  chartType,
  labels,
  values,
}: {
  chartType: "line" | "bar" | "pie"
  labels: string[]
  values: number[]
}) {
  const gradientId = useId()
  const maxValue = Math.max(1, ...values.map((v) => Math.abs(v)))
  if (chartType === "pie") {
    return <PieChartSvg labels={labels} values={values} gradientId={gradientId} />
  }
  if (chartType === "bar") {
    return <BarChartSvg labels={labels} values={values} maxValue={maxValue} gradientId={gradientId} />
  }
  return <LineChartSvg labels={labels} values={values} maxValue={maxValue} gradientId={gradientId} />
}
