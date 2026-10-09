"use client"

import { cn } from "@/lib/utils"
import { TOOL_MODES } from "./data"
import { useLanguage } from "./language-context"
import { useStudyMode } from "./study-mode-context"

export function ModeToggle({ disabled }: { disabled?: boolean }) {
  const { studyMode, setStudyMode } = useStudyMode()
  const { t } = useLanguage()

  return (
    <div
      role="group"
      aria-label={t("modes.aria")}
      className="ml-auto flex shrink-0 items-center rounded-full border border-border bg-secondary p-0.5"
    >
      {TOOL_MODES.map((mode) => {
        const selected = studyMode.id === mode.id
        return (
          <button
            key={mode.id}
            type="button"
            aria-pressed={selected}
            title={t(`modes.${mode.id}.description`)}
            disabled={disabled}
            onClick={() => setStudyMode(mode.id)}
            className={cn(
              "rounded-full px-2.5 py-1 text-xs font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40",
              selected ? "bg-brand-gradient text-white" : "text-muted-foreground hover:text-foreground",
            )}
          >
            {t(`modes.${mode.id}.shortLabel`)}
          </button>
        )
      })}
    </div>
  )
}
