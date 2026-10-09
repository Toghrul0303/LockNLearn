"use client"

import { useId } from "react"
import { Box, Calculator, PieChart, TrendingUp } from "lucide-react"
import { type Components } from "react-markdown"
import { mathRemarkPlugins, mathRehypePlugins, texifyDeskMath } from "@/lib/markdown-math"
import { MathMarkdown } from "./math-markdown"
import { type CalculationDeskUpdate, type ChartDeskUpdate, type DeskItem, type DeskUpdate, type DiagramDeskUpdate, type DiagramElement } from "./desk-context"

const W = 520
const H = 220
const PAD = 28

/** Renders the live chart/calculation payload pushed from the AI backend. */
export function DynamicDeskWidget({ update }: { update: DeskUpdate | DeskItem }) {
  if (update.type === "calculation") {
    return <CalculationCard update={update} />
  }
  if (update.type === "diagram") {
    return <DiagramCard update={update} />
  }
  return <ChartCard update={update} />
}

// `math_worker`'s `steps`/`summary`/`value` strings are now REQUIRED to use
// pure LaTeX (MATH_WORKER_SYSTEM_PROMPT's "CRITICAL MATH FORMATTING RULES")
// instead of plain text math — so, like the Chat pane (chat-pane.tsx), the
// Desk card must run them through the same remark-math/rehype-mathjax SVG
// pipeline or the raw "$...$"/"$$...$$" syntax would leak onto the card as
// literal text. `p` is overridden to a `<span>` (rather than chat-pane's
// block-level `<p>`) so short one-line steps/results stay inline within
// their existing flex/centered layouts instead of forcing an extra block.
const inlineMathComponents: Components = {
  p: ({ children }) => <span>{children}</span>,
}

/** Wraps a single AI-authored string (a `step`, `summary`, or `value`) in
 * the shared MathJax SVG renderer. Marked `.math-inline` so `globals.css` can
 * keep Desk formulas on the text baseline. */
function formatDeskValue(value: unknown): string {
  if (value == null) return ""
  if (typeof value === "string") return value
  if (typeof value === "number" || typeof value === "boolean") return String(value)
  if (typeof value === "object") {
    const obj = value as Record<string, unknown>
    for (const key of ["value", "result", "text", "display", "formatted"]) {
      if (key in obj && obj[key] != null && typeof obj[key] !== "object") {
        return String(obj[key])
      }
    }
    try {
      return JSON.stringify(value)
    } catch {
      return ""
    }
  }
  return String(value)
}

function RichText({ text, className }: { text: string; className?: string }) {
  const source = texifyDeskMath(text)
  return (
    <MathMarkdown
      as="span"
      className={className ? `math-inline ${className}` : "math-inline"}
      remarkPlugins={mathRemarkPlugins}
      rehypePlugins={mathRehypePlugins}
      components={inlineMathComponents}
    >
      {source}
    </MathMarkdown>
  )
}

/** Dedicated calculation card. Header title is stamped onto the DeskItem
 * at push time so it stays fixed when the student starts another question. */
function CalculationCard({ update }: { update: CalculationDeskUpdate & Partial<DeskItem> }) {
  const steps = update.steps ?? []
  const headerTitle = update.headerTitle?.trim() || "Math & Calculation"
  const headerDescription = update.headerDescription?.trim()

  return (
    <div
      data-highlight-source=""
      className="desk-artifact flex h-full flex-col overflow-y-auto rounded-2xl border border-border bg-card/80 p-4 shadow-sm backdrop-blur-sm"
    >
      <div className="mb-3 flex shrink-0 items-start gap-2">
        <span className="bg-brand-gradient mt-0.5 grid size-8 shrink-0 place-items-center rounded-lg text-white">
          <Calculator className="size-4" aria-hidden="true" />
        </span>
        <div className="min-w-0">
          <h3 className="font-display text-sm font-semibold text-balance">{headerTitle}</h3>
          {headerDescription ? (
            <p className="mt-0.5 line-clamp-2 text-xs leading-relaxed text-muted-foreground">
              {headerDescription}
            </p>
          ) : null}
        </div>
      </div>

      {steps.length > 0 && (
        <ol className="mb-3 space-y-1.5 rounded-xl bg-secondary/40 p-3 text-xs leading-relaxed text-foreground">
          {steps.map((step, i) => (
            <li key={i} className="flex gap-2">
              <span className="shrink-0 font-semibold text-primary">{i + 1}.</span>
              <RichText text={step} className="text-pretty" />
            </li>
          ))}
        </ol>
      )}

      <div className="flex flex-col items-center justify-center gap-2 rounded-xl bg-secondary/50 p-5 text-center">
        <span className="text-[10px] font-medium tracking-wide text-muted-foreground uppercase">
          Result
        </span>
        <RichText text={formatDeskValue(update.value)} className="text-3xl font-semibold tracking-tight text-foreground" />
        {update.summary && <RichText text={update.summary} className="max-w-prose text-sm leading-relaxed text-muted-foreground text-pretty" />}
      </div>
    </div>
  )
}

function ChartCard({ update }: { update: ChartDeskUpdate }) {
  const gradientId = useId()
  const values = update.values ?? []
  const labels = update.labels ?? []
  const maxValue = Math.max(1, ...values.map((v) => Math.abs(v)))

  return (
    <div
      data-highlight-source=""
      className="desk-artifact flex h-full flex-col rounded-2xl border border-border bg-card/80 p-4 shadow-sm backdrop-blur-sm"
    >
      <div className="mb-3 flex items-center gap-2">
        <span className="bg-brand-gradient grid size-8 place-items-center rounded-lg text-white">
          {update.chart_type === "pie" ? (
            <PieChart className="size-4" aria-hidden="true" />
          ) : (
            <TrendingUp className="size-4" aria-hidden="true" />
          )}
        </span>
        <h3 className="font-display text-sm font-semibold text-balance">{update.title || "Chart"}</h3>
      </div>

      <div className="min-h-0 flex-1">
        {update.chart_type === "pie" ? (
          <PieChartSvg labels={labels} values={values} gradientId={gradientId} />
        ) : update.chart_type === "bar" ? (
          <BarChartSvg labels={labels} values={values} maxValue={maxValue} gradientId={gradientId} />
        ) : (
          <LineChartSvg labels={labels} values={values} maxValue={maxValue} gradientId={gradientId} />
        )}
      </div>
    </div>
  )
}

function scaleX(i: number, count: number) {
  if (count <= 1) return W / 2
  return PAD + (i / (count - 1)) * (W - PAD * 2)
}
function scaleY(v: number, maxValue: number) {
  return H - PAD - (Math.abs(v) / maxValue) * (H - PAD * 2)
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

const PIE_COLORS = [
  "var(--brand-red)",
  "var(--brand-purple)",
  "var(--brand)",
  "color-mix(in oklab, var(--brand-red) 60%, var(--brand-purple))",
  "color-mix(in oklab, var(--brand) 60%, var(--brand-purple))",
]

function PieChartSvg({
  labels,
  values,
  gradientId,
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
        <title>{`Pie chart: ${gradientId}`}</title>
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

function EmptyState() {
  return (
    <div className="grid h-full place-items-center text-sm text-muted-foreground">
      No chart data returned for this result.
    </div>
  )
}

/** Minimal line-art SVG for physics/math diagrams. Coordinates are SVG pixels
 * (origin top-left). Angles: 0° = +x (right), counterclockwise in math space. */
function DiagramCard({ update }: { update: DiagramDeskUpdate }) {
  const markerId = useId().replace(/:/g, "")
  const width = Math.max(160, update.width || 400)
  const height = Math.max(120, update.height || 300)
  const elements = update.elements ?? []

  return (
    <div
      data-highlight-source=""
      className="desk-artifact flex h-full flex-col rounded-2xl border border-border bg-card/80 p-4 shadow-sm backdrop-blur-sm"
    >
      <div className="mb-3 flex items-center gap-2">
        <span className="bg-brand-gradient grid size-8 place-items-center rounded-lg text-white">
          <Box className="size-4" aria-hidden="true" />
        </span>
        <h3 className="font-display text-sm font-semibold text-balance">{update.title || "Diagram"}</h3>
      </div>
      <div className="min-h-0 flex-1">
        {elements.length === 0 ? (
          <div className="grid h-full place-items-center text-sm text-muted-foreground">
            No diagram primitives returned for this result.
          </div>
        ) : (
          <svg
            viewBox={`0 0 ${width} ${height}`}
            preserveAspectRatio="xMidYMid meet"
            className="h-full w-full"
            role="img"
            aria-label={update.title || "Diagram"}
          >
            <defs>
              <marker
                id={`${markerId}-arrow`}
                viewBox="0 0 10 10"
                refX="9"
                refY="5"
                markerWidth="7"
                markerHeight="7"
                orient="auto-start-reverse"
              >
                <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--foreground)" />
              </marker>
            </defs>
            {elements.map((el, i) => (
              <DiagramPrimitive key={i} el={el} arrowId={`${markerId}-arrow`} />
            ))}
          </svg>
        )}
      </div>
      {update.summary ? (
        <RichText
          text={update.summary}
          className="mt-2 max-w-prose text-xs leading-relaxed text-muted-foreground text-pretty"
        />
      ) : null}
    </div>
  )
}

function DiagramPrimitive({ el, arrowId }: { el: DiagramElement; arrowId: string }) {
  const dashed = el.style === "dashed" ? "6 4" : undefined
  const stroke = "var(--foreground)"
  const muted = "var(--muted-foreground)"

  if (el.type === "rect") {
    const x = el.x ?? 0
    const y = el.y ?? 0
    const w = el.width ?? 0
    const h = el.height ?? 0
    return (
      <g>
        <rect
          x={x}
          y={y}
          width={w}
          height={h}
          fill="none"
          stroke={stroke}
          strokeWidth="1.5"
          strokeDasharray={dashed}
          rx="2"
        />
        {el.label ? (
          <text
            x={x + w / 2}
            y={y + h / 2}
            textAnchor="middle"
            dominantBaseline="middle"
            className="fill-foreground text-[11px]"
          >
            {el.label}
          </text>
        ) : null}
      </g>
    )
  }

  if (el.type === "line") {
    return (
      <g>
        <line
          x1={el.x1 ?? 0}
          y1={el.y1 ?? 0}
          x2={el.x2 ?? 0}
          y2={el.y2 ?? 0}
          stroke={stroke}
          strokeWidth="1.5"
          strokeDasharray={dashed}
          strokeLinecap="round"
        />
        {el.label ? (
          <DiagramLabel x={mid(el.x1, el.x2)} y={mid(el.y1, el.y2) - 8} text={el.label} />
        ) : null}
      </g>
    )
  }

  if (el.type === "vector") {
    return (
      <g>
        <line
          x1={el.x1 ?? 0}
          y1={el.y1 ?? 0}
          x2={el.x2 ?? 0}
          y2={el.y2 ?? 0}
          stroke={stroke}
          strokeWidth="1.75"
          strokeDasharray={dashed}
          strokeLinecap="round"
          markerEnd={`url(#${arrowId})`}
        />
        {el.label ? (
          <DiagramLabel x={mid(el.x1, el.x2) + 10} y={mid(el.y1, el.y2) - 6} text={el.label} />
        ) : null}
      </g>
    )
  }

  if (el.type === "circle") {
    const cx = el.cx ?? 0
    const cy = el.cy ?? 0
    return (
      <g>
        <circle
          cx={cx}
          cy={cy}
          r={el.r ?? 0}
          fill="none"
          stroke={stroke}
          strokeWidth="1.5"
          strokeDasharray={dashed}
        />
        {el.label ? (
          <text
            x={cx}
            y={cy}
            textAnchor="middle"
            dominantBaseline="middle"
            className="fill-foreground text-[11px]"
          >
            {el.label}
          </text>
        ) : null}
      </g>
    )
  }

  if (el.type === "arc") {
    const d = arcPath(el.cx ?? 0, el.cy ?? 0, el.r ?? 0, el.start_deg ?? 0, el.end_deg ?? 0)
    const midDeg = ((el.start_deg ?? 0) + arcDelta(el.start_deg ?? 0, el.end_deg ?? 0) / 2)
    const [lx, ly] = polar(el.cx ?? 0, el.cy ?? 0, (el.r ?? 0) + 12, midDeg)
    return (
      <g>
        <path d={d} fill="none" stroke={muted} strokeWidth="1.25" strokeDasharray={dashed || "5 4"} />
        {el.label ? <DiagramLabel x={lx} y={ly} text={el.label} /> : null}
      </g>
    )
  }

  if (el.type === "text") {
    return <DiagramLabel x={el.x ?? 0} y={el.y ?? 0} text={el.text || el.label || ""} />
  }

  return null
}

function DiagramLabel({ x, y, text }: { x: number; y: number; text: string }) {
  return (
    <text x={x} y={y} textAnchor="middle" className="fill-muted-foreground text-[10px]">
      {text}
    </text>
  )
}

function mid(a?: number, b?: number) {
  return ((a ?? 0) + (b ?? 0)) / 2
}

function polar(cx: number, cy: number, r: number, deg: number): [number, number] {
  const rad = (deg * Math.PI) / 180
  return [cx + r * Math.cos(rad), cy - r * Math.sin(rad)]
}

function arcDelta(startDeg: number, endDeg: number) {
  let delta = endDeg - startDeg
  while (delta <= 0) delta += 360
  while (delta > 360) delta -= 360
  return delta
}

function arcPath(cx: number, cy: number, r: number, startDeg: number, endDeg: number) {
  if (r <= 0) return ""
  const delta = arcDelta(startDeg, endDeg)
  const [x1, y1] = polar(cx, cy, r, startDeg)
  const [x2, y2] = polar(cx, cy, r, startDeg + delta)
  const large = delta > 180 ? 1 : 0
  return `M ${x1} ${y1} A ${r} ${r} 0 ${large} 0 ${x2} ${y2}`
}
