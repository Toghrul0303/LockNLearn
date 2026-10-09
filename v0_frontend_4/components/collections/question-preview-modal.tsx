"use client"

import { useEffect } from "react"
import dynamic from "next/dynamic"
import { X } from "lucide-react"
import type { QuestionBookmark } from "@/components/workspace/memory-box-context"
import { useLanguage } from "@/components/workspace/language-context"

const QuestionPreviewBoard = dynamic(
  () => import("./question-preview-board").then((mod) => mod.QuestionPreviewBoard),
  { ssr: false },
)

export function QuestionPreviewModal({
  bookmark,
  onClose,
}: {
  bookmark: QuestionBookmark
  onClose: () => void
}) {
  const { t } = useLanguage()

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose()
    }
    window.addEventListener("keydown", onKeyDown)
    return () => window.removeEventListener("keydown", onKeyDown)
  }, [onClose])

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/70 p-4 backdrop-blur-md">
      <button
        type="button"
        onClick={onClose}
        aria-label={t("collections.closePreview")}
        className="absolute left-4 top-4 z-10 flex items-center gap-2 rounded-xl bg-background/90 px-3 py-2 text-sm font-semibold text-foreground shadow-md"
      >
        <X className="size-4" aria-hidden="true" />
        {t("collections.closePreview")}
      </button>
      <div className="question-preview h-[min(85dvh,880px)] w-full max-w-3xl overflow-hidden rounded-2xl border border-border bg-background shadow-2xl [&_[data-board-action]]:hidden">
        <div className="relative h-full w-full">
          <QuestionPreviewBoard cluster={bookmark.cluster} />
        </div>
      </div>
    </div>
  )
}
