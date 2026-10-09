"use client"

import { useState, type RefObject } from "react"
import { motion, useDragControls } from "framer-motion"
import { GripVertical, Minus, Music, Pause, Play, SkipForward, X } from "lucide-react"
import { cn } from "@/lib/utils"
import { useLanguage } from "./language-context"

type PlayerState = "open" | "minimized" | "closed"

function EqBars({ playing }: { playing: boolean }) {
  if (!playing) return null
  return (
    <span className="absolute inset-0 flex items-end justify-center gap-0.5 bg-black/10 pb-1.5">
      {[0, 1, 2, 3].map((i) => (
        <span
          key={i}
          className="w-0.5 animate-pulse rounded-full bg-white/80"
          style={{
            height: `${6 + ((i * 5) % 12)}px`,
            animationDelay: `${i * 120}ms`,
          }}
        />
      ))}
    </span>
  )
}

/** Leftmost item in the shared bottom dock. Starts minimized on every page
 * load. Draggable (via its grip handle) so it can be pulled away from the
 * dock and repositioned anywhere on the Desk canvas; expanding pops the
 * full player upward as a popover anchored to wherever it currently sits. */
export function MiniPlayer({
  dragConstraints,
}: {
  dragConstraints?: RefObject<HTMLDivElement | null>
}) {
  const [playing, setPlaying] = useState(true)
  const [state, setState] = useState<PlayerState>("minimized")
  const controls = useDragControls()
  const { t } = useLanguage()

  if (state === "closed") {
    return (
      <button
        type="button"
        onClick={() => setState("minimized")}
        aria-label={t("player.show")}
        className="grid size-11 place-items-center rounded-full border border-border bg-popover/90 text-muted-foreground shadow-xl backdrop-blur-md transition-colors hover:bg-accent hover:text-foreground"
      >
        <Music className="size-5" aria-hidden="true" />
      </button>
    )
  }

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
      {state === "open" && (
        <div className="absolute bottom-full left-0 z-10 mb-3 flex items-center gap-2 rounded-2xl border border-border bg-popover/90 p-2 pr-3 shadow-xl backdrop-blur-md">
          <span
            onPointerDown={(e) => controls.start(e)}
            role="button"
            tabIndex={0}
            aria-label={t("player.drag")}
            className="grid size-6 shrink-0 cursor-grab touch-none place-items-center rounded-md text-muted-foreground transition-colors hover:bg-accent hover:text-foreground active:cursor-grabbing"
          >
            <GripVertical className="size-4" aria-hidden="true" />
          </span>

          <div className="bg-brand-gradient relative grid size-11 shrink-0 place-items-center overflow-hidden rounded-xl text-white">
            <Music className="size-5" aria-hidden="true" />
            <EqBars playing={playing} />
          </div>

          <div className="min-w-0">
            <p className="truncate text-xs font-semibold">Deep Focus · Lo-Fi</p>
            <p className="truncate text-xs text-muted-foreground">Study Beats — 2:14 / 58:00</p>
            <div className="mt-1 h-1 w-32 overflow-hidden rounded-full bg-secondary">
              <div className="bg-brand-gradient h-full w-[8%] rounded-full" />
            </div>
          </div>

          <div className="flex items-center gap-1">
            <button
              type="button"
              onClick={() => setPlaying((p) => !p)}
              aria-label={playing ? t("player.pause") : t("player.play")}
              className="bg-brand-gradient grid size-9 place-items-center rounded-full text-white transition-transform hover:scale-105"
            >
              {playing ? (
                <Pause className="size-4" fill="currentColor" aria-hidden="true" />
              ) : (
                <Play className="size-4" fill="currentColor" aria-hidden="true" />
              )}
            </button>
            <button
              type="button"
              aria-label={t("player.next")}
              className="grid size-8 place-items-center rounded-full text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
            >
              <SkipForward className="size-4" aria-hidden="true" />
            </button>
          </div>

          <div className="ml-1 flex flex-col gap-1 border-l border-border pl-2">
            <button
              type="button"
              onClick={() => setState("minimized")}
              aria-label={t("player.minimize")}
              className="grid size-6 place-items-center rounded-md text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
            >
              <Minus className="size-3.5" aria-hidden="true" />
            </button>
            <button
              type="button"
              onClick={() => setState("closed")}
              aria-label={t("player.close")}
              className="grid size-6 place-items-center rounded-md text-muted-foreground transition-colors hover:bg-destructive/15 hover:text-destructive"
            >
              <X className="size-3.5" aria-hidden="true" />
            </button>
          </div>
        </div>
      )}

      <div className="flex items-center gap-1 rounded-full border border-border bg-popover/90 p-1.5 shadow-xl backdrop-blur-md">
        <span
          onPointerDown={(e) => controls.start(e)}
          role="button"
          tabIndex={0}
          aria-label={t("player.drag")}
          className="grid size-7 shrink-0 cursor-grab touch-none place-items-center rounded-full text-muted-foreground transition-colors hover:bg-accent hover:text-foreground active:cursor-grabbing"
        >
          <GripVertical className="size-4" aria-hidden="true" />
        </span>
        <button
          type="button"
          onClick={() => setState((s) => (s === "open" ? "minimized" : "open"))}
          aria-label={state === "open" ? t("player.collapse") : t("player.expand")}
          aria-pressed={state === "open"}
          className={cn(
            "bg-brand-gradient relative grid size-8 place-items-center overflow-hidden rounded-full text-white transition-shadow",
            state === "open" && "ring-2 ring-primary/40",
          )}
        >
          <Music className="size-4" aria-hidden="true" />
        </button>
        <button
          type="button"
          onClick={() => setPlaying((p) => !p)}
          aria-label={playing ? t("player.pause") : t("player.play")}
          className="grid size-8 place-items-center rounded-full text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
        >
          {playing ? (
            <Pause className="size-3.5" fill="currentColor" aria-hidden="true" />
          ) : (
            <Play className="size-3.5" fill="currentColor" aria-hidden="true" />
          )}
        </button>
      </div>
    </motion.div>
  )
}
