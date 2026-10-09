"use client"

import { AnimatePresence, motion } from "framer-motion"
import { CheckCircle2 } from "lucide-react"
import { useMemoryBox } from "./memory-box-context"

/** Small top-centered confirmation toast shown after "Save to FormulaBox" /
 * "Save to GraphBox". Rendered once at the app root so it works on every
 * route (`/` and `/collections`) that shares `MemoryBoxProvider`. */
export function MemoryBoxToast() {
  const { toastMessage } = useMemoryBox()

  return (
    <div className="pointer-events-none fixed inset-x-0 top-4 z-[100] flex justify-center">
      <AnimatePresence>
        {toastMessage && (
          <motion.div
            initial={{ opacity: 0, y: -16, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -12, scale: 0.95 }}
            transition={{ duration: 0.25, ease: "easeOut" }}
            className="pointer-events-auto flex items-center gap-2 rounded-full border border-border bg-popover/95 px-4 py-2 text-sm font-medium text-foreground shadow-xl backdrop-blur-md"
          >
            <CheckCircle2 className="size-4 text-emerald-500" aria-hidden="true" />
            {toastMessage}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
