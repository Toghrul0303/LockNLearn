"use client"

import { useEffect, useRef, useState, type RefObject } from "react"
import { motion, useDragControls } from "framer-motion"
import { ChevronDown, GripVertical, Pause, Play, RotateCcw } from "lucide-react"
import { cn } from "@/lib/utils"
import { useLanguage } from "./language-context"

type ModeId = "focus" | "short" | "long"

const MODE_IDS: ModeId[] = ["focus", "short", "long"]

const DEFAULT_MINUTES: Record<ModeId, number> = {
  focus: 25,
  short: 5,
  long: 15,
}

function format(seconds: number) {
  const m = Math.floor(seconds / 60)
  const s = seconds % 60
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`
}

function playDing() {
  const ctx = new AudioContext()
  const now = ctx.currentTime
  const gain = ctx.createGain()
  gain.gain.setValueAtTime(0.0001, now)
  gain.gain.exponentialRampToValueAtTime(0.18, now + 0.02)
  gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.55)
  gain.connect(ctx.destination)
  ;[880, 1175].forEach((frequency, index) => {
    const osc = ctx.createOscillator()
    osc.type = "sine"
    osc.frequency.value = frequency
    osc.connect(gain)
    const start = now + index * 0.12
    osc.start(start)
    osc.stop(start + 0.28)
  })
  window.setTimeout(() => void ctx.close(), 800)
}

/** Desk timer. Collapsed view is the remaining time and play/pause. Expanding
 * edits Focus, Short Break, and Long Break minutes. The grip in the panel
 * repositions it on the canvas. */
export function PomodoroTimer({
  dragConstraints,
}: {
  dragConstraints?: RefObject<HTMLDivElement | null>
}) {
  const [expanded, setExpanded] = useState(false)
  const [mode, setMode] = useState<ModeId>("focus")
  const [minutes, setMinutes] = useState(DEFAULT_MINUTES)
  const [sessionCount, setSessionCount] = useState(0)
  const [running, setRunning] = useState(false)
  const [remaining, setRemaining] = useState(DEFAULT_MINUTES.focus * 60)
  const controls = useDragControls()
  const { t } = useLanguage()
  const modeRef = useRef(mode)
  const minutesRef = useRef(minutes)
  const sessionRef = useRef(sessionCount)
  const completedRef = useRef(false)
  modeRef.current = mode
  minutesRef.current = minutes
  sessionRef.current = sessionCount

  const activeMinutes = minutes[mode]

  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
  useEffect(() => {
    if (!running) return
    intervalRef.current = setInterval(() => {
      setRemaining((prev) => {
        if (prev <= 1) {
          completedRef.current = true
          setRunning(false)
          return 0
        }
        return prev - 1
      })
    }, 1000)
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current)
    }
  }, [running])

  useEffect(() => {
    if (!completedRef.current || remaining !== 0 || running) return
    completedRef.current = false
    playDing()
    if (modeRef.current === "focus") {
      const next = sessionRef.current + 1
      sessionRef.current = next
      setSessionCount(next)
      const breakMode: ModeId = next % 4 === 0 ? "long" : "short"
      setMode(breakMode)
      setRemaining(minutesRef.current[breakMode] * 60)
      return
    }
    setMode("focus")
    setRemaining(minutesRef.current.focus * 60)
  }, [remaining, running])

  const selectMode = (id: ModeId) => {
    setMode(id)
    setRemaining(minutes[id] * 60)
    setRunning(false)
  }

  const updateMinutes = (id: ModeId, raw: string) => {
    const parsed = Number.parseInt(raw, 10)
    if (!Number.isFinite(parsed)) return
    const next = Math.min(180, Math.max(1, parsed))
    setMinutes((prev) => ({ ...prev, [id]: next }))
    if (id === mode) {
      setRemaining(next * 60)
      setRunning(false)
    }
  }

  const reset = () => {
    setRemaining(activeMinutes * 60)
    setRunning(false)
  }

  const modeLabel = (id: ModeId) =>
    t(id === "focus" ? "pomodoro.focus" : id === "short" ? "pomodoro.short" : "pomodoro.long")

  const total = activeMinutes * 60
  const pct = total > 0 ? ((total - remaining) / total) * 100 : 0

  return (
    <motion.div
      drag
      dragControls={controls}
      dragListener={false}
      dragMomentum={false}
      dragElastic={0.08}
      dragConstraints={dragConstraints as RefObject<Element> | undefined}
      className="relative flex items-center"
    >
      {expanded && (
        <div className="absolute right-0 bottom-full z-10 mb-3 w-64 rounded-2xl border border-border bg-popover/95 shadow-2xl backdrop-blur-md">
          <div className="flex items-center gap-1.5 border-b border-border px-2 py-2">
            <span
              onPointerDown={(e) => controls.start(e)}
              role="button"
              tabIndex={0}
              aria-label={t("pomodoro.drag")}
              className="grid size-7 shrink-0 cursor-grab touch-none place-items-center rounded-md text-muted-foreground transition-colors hover:bg-accent hover:text-foreground active:cursor-grabbing"
            >
              <GripVertical className="size-4" aria-hidden="true" />
            </span>
            <span className="text-sm font-semibold">{t("pomodoro.title")}</span>
            <button
              type="button"
              onClick={() => setExpanded(false)}
              aria-label={t("pomodoro.collapse")}
              className="ml-auto grid size-7 place-items-center rounded-md text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
            >
              <ChevronDown className="size-4" aria-hidden="true" />
            </button>
          </div>

          <div className="p-3">
            <div className="mb-3 grid grid-cols-3 gap-1 rounded-xl bg-secondary p-1">
              {MODE_IDS.map((id) => (
                <button
                  key={id}
                  type="button"
                  onClick={() => selectMode(id)}
                  className={cn(
                    "rounded-lg px-1 py-1.5 text-xs font-semibold transition-colors",
                    mode === id
                      ? "bg-background text-foreground shadow-sm"
                      : "text-muted-foreground hover:text-foreground",
                  )}
                >
                  {modeLabel(id)}
                </button>
              ))}
            </div>

            <div className="mb-3 grid grid-cols-3 gap-1">
              {MODE_IDS.map((id) => (
                <label key={id} className="flex flex-col gap-1">
                  <span className="sr-only">
                    {modeLabel(id)} {t("pomodoro.minutes")}
                  </span>
                  <input
                    type="number"
                    min={1}
                    max={180}
                    inputMode="numeric"
                    aria-label={`${modeLabel(id)} ${t("pomodoro.minutes")}`}
                    value={minutes[id]}
                    onChange={(event) => updateMinutes(id, event.target.value)}
                    className="h-8 w-full rounded-lg border border-border bg-background text-center text-xs font-semibold tabular-nums outline-none focus:border-primary/50"
                  />
                </label>
              ))}
            </div>

            <div className="text-center">
              <p className="font-display text-5xl font-bold tabular-nums tracking-tight">
                {format(remaining)}
              </p>
              <p className="mt-1 text-xs text-muted-foreground">
                {t(
                  mode === "focus"
                    ? "pomodoro.focus"
                    : mode === "short"
                      ? "pomodoro.shortBreak"
                      : "pomodoro.longBreak",
                )}{" "}
                · {activeMinutes} {t("pomodoro.minutes")}
              </p>
            </div>

            <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-secondary">
              <div
                className="bg-brand-gradient h-full rounded-full transition-all duration-500 ease-out"
                style={{ width: `${pct}%` }}
              />
            </div>

            <div className="mt-3 flex items-center justify-center gap-2">
              <button
                type="button"
                onClick={() => setRunning((r) => !r)}
                aria-label={running ? t("pomodoro.pause") : t("pomodoro.start")}
                className="bg-brand-gradient ring-brand-glow flex h-10 flex-1 items-center justify-center gap-1.5 rounded-xl text-sm font-semibold text-white transition-transform hover:-translate-y-px"
              >
                {running ? (
                  <>
                    <Pause className="size-4" fill="currentColor" aria-hidden="true" />
                    {t("pomodoro.pause")}
                  </>
                ) : (
                  <>
                    <Play className="size-4" fill="currentColor" aria-hidden="true" />
                    {t("pomodoro.start")}
                  </>
                )}
              </button>
              <button
                type="button"
                onClick={reset}
                aria-label={t("pomodoro.reset")}
                className="grid size-10 place-items-center rounded-xl border border-border text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
              >
                <RotateCcw className="size-4" aria-hidden="true" />
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="flex items-center gap-1 rounded-full border border-border bg-popover/90 p-1.5 shadow-xl backdrop-blur-md">
        <span
          onPointerDown={(e) => controls.start(e)}
          role="button"
          tabIndex={0}
          aria-label={t("pomodoro.drag")}
          className="grid size-7 shrink-0 cursor-grab touch-none place-items-center rounded-full text-muted-foreground transition-colors hover:bg-accent hover:text-foreground active:cursor-grabbing"
        >
          <GripVertical className="size-4" aria-hidden="true" />
        </span>
        <button
          type="button"
          onClick={() => setExpanded((e) => !e)}
          aria-label={expanded ? t("pomodoro.collapse") : t("pomodoro.expand")}
          aria-pressed={expanded}
          className={cn(
            "rounded-full px-2.5 text-sm font-semibold tabular-nums transition-colors hover:text-primary",
            expanded && "text-primary",
          )}
        >
          {format(remaining)}
        </button>
        <button
          type="button"
          onClick={() => setRunning((r) => !r)}
          aria-label={running ? t("pomodoro.pause") : t("pomodoro.start")}
          className="bg-brand-gradient grid size-8 place-items-center rounded-full text-white transition-transform hover:scale-105"
        >
          {running ? (
            <Pause className="size-3.5" fill="currentColor" aria-hidden="true" />
          ) : (
            <Play className="size-3.5" fill="currentColor" aria-hidden="true" />
          )}
        </button>
      </div>
    </motion.div>
  )
}
