"use client"

import { File, FileText, Image as ImageIcon, X } from "lucide-react"
import { cn } from "@/lib/utils"
import type { AttachmentKind, ChatAttachment } from "./data"

export function fileKind(file: File): AttachmentKind {
  const type = file.type.toLowerCase()
  const name = file.name.toLowerCase()
  if (type === "application/pdf" || name.endsWith(".pdf")) return "pdf"
  if (type.startsWith("image/") || /\.(png|jpe?g|gif|webp|svg)$/.test(name)) {
    return "image"
  }
  return "doc"
}

export function toChatAttachment(file: File): ChatAttachment {
  return {
    name: file.name,
    mimeType: file.type || "application/octet-stream",
    kind: fileKind(file),
    size: file.size,
  }
}

function KindIcon({ kind }: { kind: AttachmentKind }) {
  const iconClass = "size-3.5 shrink-0"
  if (kind === "pdf") return <FileText className={iconClass} aria-hidden="true" />
  if (kind === "image") return <ImageIcon className={iconClass} aria-hidden="true" />
  return <File className={iconClass} aria-hidden="true" />
}

export function AttachmentChip({
  attachment,
  variant = "on-bubble",
}: {
  attachment: ChatAttachment
  variant?: "on-bubble" | "muted"
}) {
  return (
    <span
      title={attachment.name}
      className={cn(
        "inline-flex max-w-[11rem] items-center gap-1.5 rounded-lg px-2 py-1 text-[11px] font-medium leading-tight",
        variant === "on-bubble"
          ? "bg-black/10 text-white"
          : "bg-secondary/80 text-secondary-foreground",
      )}
    >
      <KindIcon kind={attachment.kind} />
      <span className="min-w-0 truncate">{attachment.name}</span>
    </span>
  )
}

export function PendingFileChip({
  file,
  onRemove,
  disabled,
}: {
  file: File
  onRemove: () => void
  disabled?: boolean
}) {
  const attachment = toChatAttachment(file)
  return (
    <div
      title={attachment.name}
      className="inline-flex max-w-full items-center gap-2 rounded-xl border border-border bg-secondary/60 px-2.5 py-1.5 text-xs"
    >
      <KindIcon kind={attachment.kind} />
      <span className="min-w-0 max-w-[11rem] truncate font-medium text-foreground">
        {attachment.name}
      </span>
      <button
        type="button"
        aria-label="Remove attachment"
        onClick={onRemove}
        disabled={disabled}
        className="grid size-5 shrink-0 place-items-center rounded-full text-muted-foreground transition-colors hover:bg-accent hover:text-foreground disabled:cursor-not-allowed disabled:opacity-50"
      >
        <X className="size-3.5" aria-hidden="true" />
      </button>
    </div>
  )
}
