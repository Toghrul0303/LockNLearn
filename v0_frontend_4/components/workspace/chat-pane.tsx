"use client"

import { createContext, useCallback, useContext, useEffect, useLayoutEffect, useMemo, useRef, useState, type MutableRefObject, type ReactNode, type RefObject } from "react"
import { createPortal } from "react-dom"
import {
  ArrowUp,
  Check,
  CheckCircle2,
  ChevronsRight,
  FunctionSquare,
  LineChart,
  MinusCircle,
  Paperclip,
  Plus,
  RotateCcw,
  Square,
  XCircle,
} from "lucide-react"
import { type Components } from "react-markdown"
import remarkGfm from "remark-gfm"
import { mathRemarkPlugins, mathRehypePlugins, texifyDeskMath } from "@/lib/markdown-math"
import { DESK_CAMERA_EVENT } from "@/lib/desk-events"
import { MathMarkdown } from "./math-markdown"
import { cn } from "@/lib/utils"
import {
  PROMPT_TEMPLATES,
  questionNumber,
  type ChatAttachment,
  type ChatMessage,
  type Chapter,
  type PromptTemplate,
  type QuestionStatus,
  type SuggestedAction,
} from "./data"
import { AttachmentChip, PendingFileChip, toChatAttachment } from "./attachment-chip"
import { useAuth } from "./auth-context"
import { useLanguage } from "./language-context"
import { uploadSessionPdf } from "@/lib/document-sync"
import { syncExplainArmedClass, useExplainMode } from "./explain-mode-context"
import { TutorAvatar } from "./tutor-avatar"
import { useTaskTracker, nextUnansweredId } from "./task-tracker-context"
import { isCanvasOp } from "./canvas/canvas-ops"
import {
  useDesk,
  isDeskUpdate,
  isActiveProblemUpdate,
  type ActiveProblem,
  type DeskUpdate,
} from "./desk-context"
import { useSession } from "./session-context"
import { ModeToggle } from "./mode-toggle"
import { useStudyMode } from "./study-mode-context"
import { useMemoryBox } from "./memory-box-context"
import { nameSavedFormula, recordStruggle, streamChatCompletion, type StudyMode } from "@/lib/api"
import {
  isLargePdf,
  LARGE_PDF_BYTES,
  parseTaskAssignment,
  parseStartConfirm,
  parseChatAction,
  buildAssignmentAck,
  buildNextQuestionPrompt,
  buildTaskStartPrompt,
  buildTaskStartDisplay,
  isInternalRoutingText,
  userFacingProblemText,
  looksAzerbaijani,
} from "@/lib/parse-task-assignment"

/** Prefer this chapter's question N, then the newest chapter that has it. */
function questionWithNumber(chapters: Chapter[], number: number, activeQuestionId: string | null) {
  const activeChapter = chapters.find((chapter) =>
    chapter.questions.some((question) => question.id === activeQuestionId),
  )
  const inActive = activeChapter?.questions.find((question) => questionNumber(question) === number)
  if (inActive) return inActive
  for (let index = chapters.length - 1; index >= 0; index--) {
    const found = chapters[index].questions.find((question) => questionNumber(question) === number)
    if (found) return found
  }
  return undefined
}

/** Collapses a string down to its first `maxWords` words for a clean, scannable title. */
function truncateToWords(text: string, maxWords: number): string {
  const words = text.trim().split(/\s+/).filter(Boolean)
  if (words.length === 0) return ""
  if (words.length <= maxWords) return words.join(" ")
  return `${words.slice(0, maxWords).join(" ")}…`
}

/** True for empty / "həll et" / "Bunu da həll et" / "solve this" / internal
 * `[TASK START]` routing — never use those as the Active Problem subtitle. */
function isGenericUserPrompt(query: string): boolean {
  const cleaned = query.trim().replace(/\s+/g, " ")
  if (!cleaned) return true
  if (isInternalRoutingText(cleaned)) return true
  return /^(?:(?:zəhmət\s+olmasa|please|pls)\s+)?(?:(?:bunu(?:\s+da)?|bu\s+sualı|bu\s+məsələni|this|that|it|the\s+problem)\s+)?(?:həll\s*et(?:məyi)?|solve(?:\s+(?:it|this|the\s+problem))?|hesabla|kömək(?:\s*et)?|help(?:\s+me)?)(?:\s+(?:zəhmət\s+olmasa|please|pls|də|da))?[.!?…]*$/i.test(
    cleaned,
  )
}

const DEFAULT_PROBLEM_SUBTITLE = "Sənəd üzrə məsələnin həlli və təhlili"

/** Builds the Desk's "Active problem" header — but ONLY ever called when a
 * Desk visual (chart/calculation) was actually generated this turn. Prefers
 * the chart's own AI-authored `title` (already short & on-topic); falls back
 * to a word-truncated summary of the raw user query for calculations.
 * Generic "həll et" prompts are never copied into `description`. */
function buildActiveProblemFromDeskUpdate(
  update: DeskUpdate,
  rawQuery: string,
  fileName?: string | null,
): ActiveProblem {
  const trimmedQuery = rawQuery.trim()
  const generic = isGenericUserPrompt(trimmedQuery)
  const visualTitle =
    (update.type === "chart" || update.type === "diagram") && update.title
      ? truncateToWords(update.title, 7)
      : ""
  const smartTitle = visualTitle || (generic ? "" : truncateToWords(rawQuery, 7))
  const summary =
    "summary" in update && typeof update.summary === "string" ? update.summary.trim() : ""

  const description = generic
    ? summary || DEFAULT_PROBLEM_SUBTITLE
    : userFacingProblemText(trimmedQuery)
      ? fileName
        ? `${userFacingProblemText(trimmedQuery)} (with attachment: ${fileName})`
        : userFacingProblemText(trimmedQuery)
      : smartTitle || "Workspace visual generated."

  return {
    title: userFacingProblemText(smartTitle) || (fileName ? fileName.replace(/\.[^.]+$/, "") : "New Workspace result"),
    description: userFacingProblemText(description) || DEFAULT_PROBLEM_SUBTITLE,
  }
}

// Tailwind-styled renderers so streamed Markdown (headers, lists, bold,
// code, links) gets real formatted elements instead of relying on a
// typography plugin that isn't installed in this project.
const CHAT_REMARK_PLUGINS = [...mathRemarkPlugins, remarkGfm]
const CHAT_REHYPE_PLUGINS = mathRehypePlugins

const markdownComponents: Components = {
  p: ({ children }) => <p className="mb-2 leading-relaxed last:mb-0">{children}</p>,
  strong: ({ children }) => <strong className="font-semibold text-foreground">{children}</strong>,
  em: ({ children }) => <em className="italic">{children}</em>,
  ul: ({ children }) => <ul className="mb-2 list-disc space-y-1 pl-5 last:mb-0">{children}</ul>,
  ol: ({ children }) => <ol className="mb-2 list-decimal space-y-1 pl-5 last:mb-0">{children}</ol>,
  li: ({ children }) => <li className="leading-relaxed">{children}</li>,
  h1: ({ children }) => (
    <h1 className="mt-3 mb-1.5 font-display text-base font-semibold text-foreground first:mt-0">{children}</h1>
  ),
  h2: ({ children }) => (
    <h2 className="mt-3 mb-1.5 font-display text-[0.95rem] font-semibold text-foreground first:mt-0">{children}</h2>
  ),
  h3: ({ children }) => (
    <h3 className="mt-2.5 mb-1 font-display text-sm font-semibold text-foreground first:mt-0">{children}</h3>
  ),
  a: ({ children, href }) => (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="text-primary underline underline-offset-2 hover:text-primary/80"
    >
      {children}
    </a>
  ),
  blockquote: ({ children }) => (
    <blockquote className="mb-2 border-l-2 border-border pl-3 text-muted-foreground italic last:mb-0">
      {children}
    </blockquote>
  ),
  code: ({ className, children, ...props }) => {
    const isBlock = /language-/.test(className ?? "")
    if (isBlock) {
      return (
        <code className={cn("font-mono text-xs", className)} {...props}>
          {children}
        </code>
      )
    }
    return (
      <code className="rounded bg-secondary px-1 py-0.5 font-mono text-[0.8em]" {...props}>
        {children}
      </code>
    )
  },
  pre: ({ children }) => (
    <pre className="mb-2 overflow-x-auto rounded-lg bg-secondary/70 p-2.5 last:mb-0">{children}</pre>
  ),
  table: ({ children }) => (
    <div className="mb-2 overflow-x-auto last:mb-0">
      <table className="w-full border-collapse text-left text-xs">{children}</table>
    </div>
  ),
  th: ({ children }) => (
    <th className="border border-border bg-secondary/50 px-2 py-1 font-semibold">{children}</th>
  ),
  td: ({ children }) => <td className="border border-border px-2 py-1">{children}</td>,
}

const ChatActionMessageIdContext = createContext<string | undefined>(undefined)

function ActionLink({
  href,
  disabled,
  onAction,
  children,
}: {
  href: string
  disabled?: boolean
  onAction: (href: string, messageId?: string) => void
  children: ReactNode
}) {
  const messageId = useContext(ChatActionMessageIdContext)
  const [used, setUsed] = useState(false)
  return (
    <button
      type="button"
      disabled={disabled || used}
      onClick={(event) => {
        event.preventDefault()
        if (disabled || used) return
        setUsed(true)
        onAction(href, messageId)
      }}
      className="mx-0.5 inline-flex translate-y-px items-center rounded-full border border-primary/25 bg-primary/10 px-2 py-0.5 text-xs font-semibold text-primary transition-colors hover:bg-primary/15 disabled:cursor-not-allowed disabled:opacity-50"
    >
      {children}
    </button>
  )
}

function createMarkdownComponents(
  onActionRef: MutableRefObject<((href: string, messageId?: string) => void) | undefined>,
  sendingRef: MutableRefObject<boolean>,
): Components {
  return {
    ...markdownComponents,
    a: ({ children, href }) => {
      if (parseChatAction(href)) {
        return (
          <ActionLink
            href={href ?? ""}
            disabled={sendingRef.current}
            onAction={(nextHref, messageId) => {
              if (sendingRef.current) return
              onActionRef.current?.(nextHref, messageId)
            }}
          >
            {children}
          </ActionLink>
        )
      }
      return (
        <a
          href={href}
          target="_blank"
          rel="noopener noreferrer"
          className="text-primary underline underline-offset-2 hover:text-primary/80"
        >
          {children}
        </a>
      )
    },
  }
}

type HighlightAsk = {
  text: string
  x: number
  y: number
  placement: "above" | "below"
  shapeId?: string
  formulaTex?: string
}

const HIGHLIGHT_ASK_MAX_CHARS = 1200
const HIGHLIGHT_ASK_MIN_CHARS = 2
const HIGHLIGHT_ASK_TOOLBAR_W = 340
const EXPLAIN_GRACE_MS = 700
const EXPLAIN_TIP_MS = 3000
const HIGHLIGHT_ASK_TOOLBAR_H = 42

const HIGHLIGHT_HOST_SELECTOR =
  "[data-ai-message], [data-highlight-source], .desk-artifact, .board-node"

function nodeElement(node: Node | null): Element | null {
  if (!node) return null
  return node instanceof Element ? node : node.parentElement
}

function closestHighlightHost(node: Node | null): Element | null {
  return nodeElement(node)?.closest(HIGHLIGHT_HOST_SELECTOR) ?? null
}

function closestMjx(node: Node | null): Element | null {
  return nodeElement(node)?.closest("mjx-container, .MathJax") ?? null
}

function isAbortError(error: unknown): boolean {
  return (
    (error instanceof DOMException && error.name === "AbortError") ||
    (typeof error === "object" &&
      error !== null &&
      "name" in error &&
      (error as { name: string }).name === "AbortError")
  )
}

type MathJaxItemLike = {
  math?: string
  display?: boolean
  typesetRoot?: Node | null
}

function mathJaxItems(): MathJaxItemLike[] {
  try {
    const doc = (
      window as Window & {
        MathJax?: { startup?: { document?: { math?: Iterable<MathJaxItemLike> } } }
      }
    ).MathJax?.startup?.document
    const list = doc?.math
    if (!list) return []
    return Array.from(list)
  } catch {
    return []
  }
}

function texFromMathJaxItem(el: Element): { tex: string; display: boolean } | null {
  const container = el.closest("mjx-container, .MathJax") ?? el
  for (const item of mathJaxItems()) {
    const root = item.typesetRoot
    if (!(root instanceof Node)) continue
    const same =
      root === container ||
      container.contains(root) ||
      (root instanceof Element && root.contains(container))
    if (!same) continue
    const tex = item.math?.trim()
    if (!tex) continue
    return { tex, display: Boolean(item.display) }
  }
  return null
}

function texFromMjxContainer(el: Element): string | null {
  const container = el.closest("mjx-container, .MathJax") ?? el
  const ann =
    container.querySelector('annotation[encoding="application/x-tex"]') ??
    container.querySelector('annotation[encoding="application/x-latex"]')
  const annotated = ann?.textContent?.trim()
  const fromItem = annotated ? null : texFromMathJaxItem(container)
  const aria =
    container.getAttribute("aria-label")?.trim() ||
    container.querySelector("svg[aria-label]")?.getAttribute("aria-label")?.trim() ||
    container.querySelector("[aria-label]")?.getAttribute("aria-label")?.trim()
  const raw = annotated || fromItem?.tex || aria
  if (!raw) return null
  if (aria && !annotated && !fromItem) return raw
  const display = annotated
    ? container.getAttribute("display") === "true"
    : Boolean(fromItem?.display) || container.getAttribute("display") === "true"
  const trimmed = raw.trim()
  if (trimmed.startsWith("$$") && trimmed.endsWith("$$")) return trimmed
  if (trimmed.startsWith("$") && trimmed.endsWith("$")) {
    return display ? `$$${trimmed.slice(1, -1)}$$` : trimmed
  }
  return display ? `$$${trimmed}$$` : `$${trimmed}$`
}

function unionClientRects(rects: DOMRectList | DOMRect[]): DOMRect | null {
  let left = Infinity
  let top = Infinity
  let right = -Infinity
  let bottom = -Infinity
  const list = Array.from(rects)
  for (const r of list) {
    if (r.width === 0 && r.height === 0) continue
    left = Math.min(left, r.left)
    top = Math.min(top, r.top)
    right = Math.max(right, r.right)
    bottom = Math.max(bottom, r.bottom)
  }
  if (!Number.isFinite(left)) return null
  return new DOMRect(left, top, right - left, bottom - top)
}

function selectionViewportRect(range: Range, host: Element): DOMRect | null {
  const direct = range.getBoundingClientRect()
  if (direct.width > 0 || direct.height > 0) return direct

  const union = unionClientRects(range.getClientRects())
  if (union) return union

  const mjx =
    closestMjx(range.commonAncestorContainer) ??
    closestMjx(range.startContainer) ??
    host.querySelector("mjx-container, .MathJax")
  if (mjx) {
    const mjxRect = mjx.getBoundingClientRect()
    if (mjxRect.width > 0 || mjxRect.height > 0) return mjxRect
  }

  const hostRect = host.getBoundingClientRect()
  if (hostRect.width > 0 || hostRect.height > 0) return hostRect
  return null
}

function highlightFromTextAndRect(
  text: string,
  rect: DOMRect,
  shapeId?: string,
  formulaTex?: string,
): HighlightAsk | null {
  const trimmed = text.replace(/\s+/g, " ").trim()
  if (trimmed.length < HIGHLIGHT_ASK_MIN_CHARS) return null
  const quoted =
    trimmed.length > HIGHLIGHT_ASK_MAX_CHARS
      ? `${trimmed.slice(0, HIGHLIGHT_ASK_MAX_CHARS).trim()}…`
      : trimmed

  const pad = 12
  let x = rect.left + rect.width / 2
  const half = HIGHLIGHT_ASK_TOOLBAR_W / 2
  x = Math.min(window.innerWidth - pad - half, Math.max(pad + half, x))
  const placement: HighlightAsk["placement"] =
    rect.top >= HIGHLIGHT_ASK_TOOLBAR_H + pad ? "above" : "below"
  const y = placement === "above" ? rect.top - 8 : rect.bottom + 8
  return { text: quoted, x, y, placement, shapeId, formulaTex }
}

function textPortionInRange(text: Text, range: Range): string {
  if (!range.intersectsNode(text)) return ""
  const start = range.startContainer === text ? range.startOffset : 0
  const end = range.endContainer === text ? range.endOffset : text.data.length
  return text.data.slice(start, end)
}

function nodeIntersectsRange(node: Node, range: Range): boolean {
  try {
    return range.intersectsNode(node)
  } catch {
    return false
  }
}

function extractHighlightText(range: Range, host: Element): string {
  const parts: string[] = []
  const walker = document.createTreeWalker(host, NodeFilter.SHOW_ELEMENT | NodeFilter.SHOW_TEXT)
  let skipDescendantsOf: Element | null = null
  let node: Node | null = walker.nextNode()
  while (node) {
    if (skipDescendantsOf) {
      if (skipDescendantsOf.contains(node) && node !== skipDescendantsOf) {
        node = walker.nextNode()
        continue
      }
      skipDescendantsOf = null
    }
    if (node.nodeType === Node.ELEMENT_NODE) {
      const el = node as Element
      const tag = el.tagName.toLowerCase()
      if (tag === "mjx-container" || el.classList.contains("MathJax")) {
        if (nodeIntersectsRange(el, range) || el.contains(range.commonAncestorContainer)) {
          parts.push(texFromMjxContainer(el) ?? "")
        }
        skipDescendantsOf = el
      }
    } else if (node.nodeType === Node.TEXT_NODE) {
      parts.push(textPortionInRange(node as Text, range))
    }
    node = walker.nextNode()
  }
  const mixed = parts.join("").replace(/\s+/g, " ").trim()
  if (mixed) return mixed
  const startMjx = closestMjx(range.startContainer)
  if (startMjx && host.contains(startMjx)) {
    const tex = texFromMjxContainer(startMjx)
    if (tex) return tex
  }
  return range.toString().replace(/\s+/g, " ").trim()
}

function highlightFromMjx(mjx: Element): HighlightAsk | null {
  const host = closestHighlightHost(mjx)
  if (!host) return null
  const tex = texFromMjxContainer(mjx)
  if (!tex) return null
  const shapeHost = host.closest("[data-shape-id]") ?? host
  const shapeId = shapeHost.getAttribute("data-shape-id") || undefined
  return highlightFromTextAndRect(tex, mjx.getBoundingClientRect(), shapeId, tex)
}

function formulaTexInRange(range: Range, host: Element, eventTarget?: Node | null): string | undefined {
  const containers: Element[] = []
  const add = (node: Node | null) => {
    let current: Node | null = node
    while (current) {
      const mjx = closestMjx(current)
      if (mjx && host.contains(mjx) && !containers.includes(mjx)) containers.push(mjx)
      current = current.parentNode
      if (current === host) break
    }
  }
  add(range.startContainer)
  add(range.endContainer)
  add(range.commonAncestorContainer)
  add(eventTarget ?? null)
  const sel = window.getSelection()
  add(sel?.anchorNode ?? null)
  add(sel?.focusNode ?? null)
  host.querySelectorAll("mjx-container, .MathJax").forEach((mjx) => {
    const inside =
      mjx.contains(range.startContainer) ||
      mjx.contains(range.endContainer) ||
      mjx.contains(range.commonAncestorContainer) ||
      nodeIntersectsRange(mjx, range)
    if (inside && !containers.includes(mjx)) containers.push(mjx)
  })
  const parts = containers
    .map((mjx) => texFromMjxContainer(mjx))
    .filter((tex): tex is string => Boolean(tex))
  const joined = parts.join(" ").trim()
  return joined || undefined
}

function clearMathSelectionPaint() {
  document.querySelectorAll("[data-math-selected]").forEach((el) => {
    el.removeAttribute("data-math-selected")
  })
}

/** SVG glyphs ignore `::selection`; paint the mjx-container instead. */
function paintMathSelection(eventTarget?: Node | null) {
  clearMathSelectionPaint()
  const sel = window.getSelection()
  if (sel && !sel.isCollapsed && sel.rangeCount > 0) {
    const range = sel.getRangeAt(0)
    const host =
      closestHighlightHost(range.commonAncestorContainer) ??
      closestHighlightHost(sel.anchorNode) ??
      closestHighlightHost(sel.focusNode) ??
      closestHighlightHost(eventTarget ?? null)
    if (!host) return
    host.querySelectorAll("mjx-container, .MathJax").forEach((el) => {
      if (nodeIntersectsRange(el, range) || el.contains(range.commonAncestorContainer)) {
        el.setAttribute("data-math-selected", "")
      }
    })
    return
  }
  const mjx = closestMjx(eventTarget ?? null)
  if (mjx && closestHighlightHost(mjx)) {
    mjx.setAttribute("data-math-selected", "")
  }
}

/** True when the live selection (or a MathJax click) sits inside one AI bubble or Desk card. */
function readHighlightSelection(event?: Event): HighlightAsk | null {
  const sel = window.getSelection()
  const eventTarget = event?.target instanceof Node ? event.target : null

  if (!sel || sel.isCollapsed || sel.rangeCount === 0) {
    const mjx = closestMjx(eventTarget)
    if (!mjx) return null
    if (!closestHighlightHost(mjx)) return null
    return highlightFromMjx(mjx)
  }

  const range = sel.getRangeAt(0)
  const host =
    closestHighlightHost(range.commonAncestorContainer) ??
    closestHighlightHost(sel.anchorNode) ??
    closestHighlightHost(sel.focusNode) ??
    closestHighlightHost(eventTarget)
  if (!host) return null

  const anchorEl = nodeElement(sel.anchorNode) ?? closestMjx(sel.anchorNode)
  const focusEl = nodeElement(sel.focusNode) ?? closestMjx(sel.focusNode)
  if (!anchorEl || !focusEl) return null
  if (!host.contains(anchorEl) || !host.contains(focusEl)) return null

  const text = extractHighlightText(range, host)
  const rect = selectionViewportRect(range, host)
  if (!rect) return null
  const shapeHost = host.closest("[data-shape-id]") ?? host
  const shapeId = shapeHost.getAttribute("data-shape-id") || undefined
  return highlightFromTextAndRect(text, rect, shapeId, formulaTexInRange(range, host, eventTarget))
}

function documentPathStorageKey(threadId: string) {
  return `locknlearn.documentPath.${threadId}`
}

function readStoredDocumentPath(threadId: string) {
  try {
    return window.sessionStorage.getItem(documentPathStorageKey(threadId)) || ""
  } catch {
    return ""
  }
}

function writeStoredDocumentPath(threadId: string, path: string) {
  try {
    window.sessionStorage.setItem(documentPathStorageKey(threadId), path)
  } catch {
    // Private browsing can reject sessionStorage. The in-memory ref still covers this tab.
  }
}

function buildHighlightAskPrompt(quoted: string, source: "chat" | "desk"): string {
  const followup =
    "Explain this specific part. Do not restate the whole solution."
  if (source === "desk") {
    return `[CANVAS EXPLAIN] Regarding this part of the canvas step: "${quoted}" — ${followup}

[EXPLAIN SOURCE: desk]
Call python_code_executor once with type "calculation" and put the full explanation in the steps list. Chat must only acknowledge that the explanation was added to the workspace.`
  }
  return `[CANVAS EXPLAIN] Regarding this part of the canvas step: "${quoted}" — ${followup}

[EXPLAIN SOURCE: chat]
Answer in the chat only. Do not call python_code_executor. Do not emit canvas steps or a new desk box.`
}

const PIPELINE_STATUSES = new Set([
  "Indexing document (large textbooks may take up to 1 minute)...",
  "Locating target question in document...",
  "Page located, analyzing visual context...",
  "Executing mathematical engine & formulating solution...",
  "Rendering output to whiteboard...",
])

export function ChatPane({
  collapsed,
  chatWidth,
  onChatWidthChange,
  onChatWidthCommit,
  onToggle,
}: {
  collapsed: boolean
  chatWidth: number
  onChatWidthChange: (width: number) => void
  onChatWidthCommit: (width: number) => void
  onToggle: () => void
}) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState("")
  const [saved, setSaved] = useState<Record<string, boolean>>({})
  const [attachedFile, setAttachedFile] = useState<File | null>(null)
  const [isSending, setIsSending] = useState(false)
  const [pipelineStatus, setPipelineStatus] = useState("")
  const [highlightAsk, setHighlightAsk] = useState<HighlightAsk | null>(null)
  const [explainTip, setExplainTip] = useState<{ x: number; y: number; placement: "above" | "below" } | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const highlightToolbarRef = useRef<HTMLDivElement>(null)
  const explainTipRef = useRef<HTMLDivElement>(null)
  const highlightAskRef = useRef(highlightAsk)
  const graceTimerRef = useRef<number | null>(null)
  const highlightDismissRef = useRef<number | null>(null)
  const tipTimerRef = useRef<number | null>(null)
  const captureTimerRef = useRef<number | null>(null)
  highlightAskRef.current = highlightAsk
  const isSendingRef = useRef(isSending)
  const pointerDownRef = useRef(false)
  const abortRef = useRef<AbortController | null>(null)
  const streamingAiIdRef = useRef<string | null>(null)
  const pendingSolveRef = useRef<{ requestId: string; shapeId?: string } | null>(null)
  /** Board-question card currently anchoring the active Socratic dialogue. */
  const socraticAnchorRef = useRef<string | null>(null)
  isSendingRef.current = isSending
  const {
    activeQuestion,
    activeQuestionId,
    chapters,
    setStatus,
    assignModule,
    startQuestion,
    setOnStartQuestion,
    setOnQuestionStatus,
    setAutoReviewed,
  } = useTaskTracker()
  const chaptersRef = useRef(chapters)
  chaptersRef.current = chapters
  /** Holds the still-unclear note until the user marks this question Needs Review again. */
  const reviewHintHold = useRef<string | null>(null)
  const activeQuestionIdRef = useRef(activeQuestionId)
  activeQuestionIdRef.current = activeQuestionId
  const onChatActionRef = useRef<(href: string, messageId?: string) => void>(() => {})
  const actionMarkdown = useMemo(
    () => createMarkdownComponents(onChatActionRef, isSendingRef),
    [],
  )
  const {
    pushDeskUpdate,
    applyCanvasOp,
    getActiveQuestionId,
    registerSolveHandler,
    completeSolve,
    setActiveProblem,
    deskItems,
    activeProblem,
    stampHeadersOnItems,
    focusTrackerQuestion,
    readTrackerQuestionPrompt,
  } = useDesk()
  const { armed } = useExplainMode()
  const armedRef = useRef(armed)
  armedRef.current = armed
  const { threadId, isDraft, commitSessionTitle, getSession, patchSession, sessionsReady } = useSession()
  const documentPathRef = useRef("")
  useEffect(() => {
    documentPathRef.current = readStoredDocumentPath(threadId)
  }, [threadId])
  const { studyMode } = useStudyMode()
  const { locale, t } = useLanguage()
  const appendReviewHint = useCallback((prev: ChatMessage[], questionId: string) => {
    const hintId = `review-hint:${questionId}`
    if (prev.some((message) => message.id === hintId)) return prev
    return [
      ...prev,
      {
        id: hintId,
        role: "ai" as const,
        content: t("chat.reviewStillUnclear"),
      },
    ]
  }, [t])
  const { user, session: authSession } = useAuth()
  const { saveFormula, renameFormula, saveGraph, saveStruggle } = useMemoryBox()
  const prevThreadId = useRef(threadId)
  const getSessionRef = useRef(getSession)
  getSessionRef.current = getSession
  const persistReady = useRef(false)
  const pendingFileRef = useRef<File | null>(null)
  const attachedFileRef = useRef<File | null>(null)
  attachedFileRef.current = attachedFile
  const [pendingFileBound, setPendingFileBound] = useState(false)
  const [pendingSelect, setPendingSelect] = useState<{ start: number; end: number } | null>(
    null,
  )
  const [templatesOpen, setTemplatesOpen] = useState(false)
  const templatesMenuRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    })
  }, [messages])

  useLayoutEffect(() => {
    const el = inputRef.current
    if (!el) return
    // Collapse first so deleting/clearing can shrink; `auto` alone won't
    // drop below the previous explicit pixel height in some browsers.
    el.style.height = "0px"
    const cap = Math.round(window.innerHeight * 0.4)
    const next = Math.min(Math.max(el.scrollHeight, 36), cap)
    el.style.height = `${next}px`
    el.style.overflowY = el.scrollHeight > cap ? "auto" : "hidden"
  }, [input])

  useLayoutEffect(() => {
    if (!pendingSelect) return
    const el = inputRef.current
    if (!el) return
    el.focus()
    el.setSelectionRange(pendingSelect.start, pendingSelect.end)
    setPendingSelect(null)
  }, [pendingSelect, input])

  // Hydrate this pane from the session record when the thread changes.
  // A first-mount load also runs (prev === threadId is allowed) so a restored
  // session is not stuck on an empty transcript.
  useEffect(() => {
    if (!sessionsReady) return
    const switched = prevThreadId.current !== threadId
    prevThreadId.current = threadId
    persistReady.current = false
    const rec = getSessionRef.current(threadId)
    setMessages(rec?.messages ?? [])
    if (switched) {
      abortRef.current?.abort()
      abortRef.current = null
      streamingAiIdRef.current = null
      setInput("")
      setAttachedFile(null)
      setSaved({})
      setIsSending(false)
      cancelHighlightDismiss()
      clearMathSelectionPaint()
      setHighlightAsk(null)
      pendingFileRef.current = null
      setPendingFileBound(false)
      setPendingSelect(null)
      setTemplatesOpen(false)
    }
    const enable = window.setTimeout(() => {
      persistReady.current = true
    }, 0)
    return () => window.clearTimeout(enable)
  }, [threadId, sessionsReady])

  useEffect(() => {
    if (!sessionsReady || isDraft || !persistReady.current) return
    const timer = window.setTimeout(() => {
      patchSession(threadId, { messages })
    }, 250)
    return () => window.clearTimeout(timer)
  }, [messages, threadId, isDraft, patchSession, sessionsReady])

  const attachImageFile = (file: File) => {
    const type = file.type || "image/png"
    const ext = (type.split("/")[1] || "png").replace("jpeg", "jpg")
    const named =
      file.name && !/^image\.(png|jpe?g|webp|gif)$/i.test(file.name)
        ? file
        : new File([file], `screenshot.${ext}`, { type })
    pendingFileRef.current = null
    setPendingFileBound(false)
    setAttachedFile(named)
  }

  const markMessageAborted = (id: string) => {
    setMessages((prev) =>
      prev.map((m) =>
        m.id === id
          ? { ...m, aborted: true, content: "", formula: undefined, suggestedActions: undefined }
          : m,
      ),
    )
  }

  const clearGraceTimer = () => {
    if (graceTimerRef.current != null) {
      window.clearTimeout(graceTimerRef.current)
      graceTimerRef.current = null
    }
  }

  const clearTipTimer = () => {
    if (tipTimerRef.current != null) {
      window.clearTimeout(tipTimerRef.current)
      tipTimerRef.current = null
    }
  }

  const clearCaptureTimer = () => {
    if (captureTimerRef.current != null) {
      window.clearTimeout(captureTimerRef.current)
      captureTimerRef.current = null
    }
  }

  const cancelHighlightDismiss = () => {
    if (highlightDismissRef.current != null) {
      window.clearTimeout(highlightDismissRef.current)
      highlightDismissRef.current = null
    }
  }

  const scheduleHighlightDismiss = () => {
    cancelHighlightDismiss()
    highlightDismissRef.current = window.setTimeout(() => {
      highlightDismissRef.current = null
      clearMathSelectionPaint()
      setHighlightAsk(null)
    }, EXPLAIN_GRACE_MS)
  }

  const dismissExplainUi = () => {
    cancelHighlightDismiss()
    clearGraceTimer()
    clearTipTimer()
    clearMathSelectionPaint()
    setExplainTip(null)
    setHighlightAsk(null)
  }

  const showExplainTip = (tip: { x: number; y: number; placement: "above" | "below" }) => {
    clearTipTimer()
    setExplainTip(tip)
    tipTimerRef.current = window.setTimeout(() => {
      tipTimerRef.current = null
      setExplainTip(null)
    }, EXPLAIN_TIP_MS)
  }

  const captureHighlightAsk = (event?: Event) => {
    if (isSendingRef.current) return
    const next = readHighlightSelection(event)
    if (!next) return
    const alt = event instanceof MouseEvent && event.altKey
    if (armedRef.current || alt) {
      clearTipTimer()
      setExplainTip(null)
      clearGraceTimer()
      cancelHighlightDismiss()
      const eventTarget = event?.target instanceof Node ? event.target : null
      paintMathSelection(eventTarget)
      setHighlightAsk(next)
      return
    }
    clearGraceTimer()
    clearMathSelectionPaint()
    setHighlightAsk(null)
    showExplainTip({ x: next.x, y: next.y, placement: next.placement })
  }

  const scheduleCapture = (event: Event) => {
    clearCaptureTimer()
    captureTimerRef.current = window.setTimeout(() => {
      captureTimerRef.current = null
      captureHighlightAsk(event)
    }, 0)
  }

  useEffect(() => {
    const onPointerDown = (e: PointerEvent) => {
      pointerDownRef.current = true
      const target = e.target as Node | null
      if (highlightToolbarRef.current?.contains(target)) {
        cancelHighlightDismiss()
        return
      }
      if (explainTipRef.current?.contains(target)) return
      scheduleHighlightDismiss()
    }
    const onPointerUp = (e: PointerEvent) => {
      pointerDownRef.current = false
      scheduleCapture(e)
    }
    const onMouseUp = (e: MouseEvent) => {
      pointerDownRef.current = false
      scheduleCapture(e)
    }
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") dismissExplainUi()
    }
    const onSelectionChange = () => {
      if (!highlightAskRef.current || pointerDownRef.current || isSendingRef.current) return
      paintMathSelection()
    }
    document.addEventListener("pointerdown", onPointerDown)
    document.addEventListener("pointerup", onPointerUp)
    document.addEventListener("mouseup", onMouseUp)
    document.addEventListener("keydown", onKeyDown)
    document.addEventListener("selectionchange", onSelectionChange)
    const onScroll = () => scheduleHighlightDismiss()
    document.addEventListener("scroll", onScroll, true)
    const onDeskCamera = () => scheduleHighlightDismiss()
    window.addEventListener(DESK_CAMERA_EVENT, onDeskCamera)
    return () => {
      document.removeEventListener("pointerdown", onPointerDown)
      document.removeEventListener("pointerup", onPointerUp)
      document.removeEventListener("mouseup", onMouseUp)
      document.removeEventListener("keydown", onKeyDown)
      document.removeEventListener("selectionchange", onSelectionChange)
      document.removeEventListener("scroll", onScroll, true)
      window.removeEventListener(DESK_CAMERA_EVENT, onDeskCamera)
      clearGraceTimer()
      cancelHighlightDismiss()
      clearTipTimer()
      clearCaptureTimer()
      syncExplainArmedClass(false)
    }
  }, [])

  useEffect(() => {
    syncExplainArmedClass(highlightAsk != null)
    if (!highlightAsk) {
      clearMathSelectionPaint()
      clearGraceTimer()
      return
    }
    const onOver = (e: PointerEvent) => {
      const node = e.target instanceof Node ? e.target : null
      if (!node) return
      if (highlightToolbarRef.current?.contains(node) || closestHighlightHost(node)) {
        clearGraceTimer()
      }
    }
    const onOut = (e: PointerEvent) => {
      const from = e.target instanceof Node ? e.target : null
      const next = e.relatedTarget instanceof Node ? e.relatedTarget : null
      if (!from) return
      const toolbar = highlightToolbarRef.current
      const fromInChip = Boolean(toolbar?.contains(from))
      const fromInHost = Boolean(closestHighlightHost(from))
      if (!fromInChip && !fromInHost) return
      const nextInChip = Boolean(next && toolbar?.contains(next))
      const nextInHost = Boolean(next && closestHighlightHost(next))
      if (nextInChip || nextInHost) {
        clearGraceTimer()
        return
      }
      clearGraceTimer()
      graceTimerRef.current = window.setTimeout(() => {
        graceTimerRef.current = null
        clearMathSelectionPaint()
        setHighlightAsk(null)
      }, EXPLAIN_GRACE_MS)
    }
    document.addEventListener("pointerover", onOver)
    document.addEventListener("pointerout", onOut)
    return () => {
      document.removeEventListener("pointerover", onOver)
      document.removeEventListener("pointerout", onOut)
      clearGraceTimer()
    }
  }, [highlightAsk])

  useEffect(() => {
    const el = scrollRef.current
    if (!el || !highlightAsk) return
    const onScroll = () => scheduleHighlightDismiss()
    el.addEventListener("scroll", onScroll, { passive: true })
    return () => el.removeEventListener("scroll", onScroll)
  }, [highlightAsk])

  useEffect(() => {
    if (!templatesOpen) return
    const onPointerDown = (e: PointerEvent) => {
      if (templatesMenuRef.current?.contains(e.target as Node)) return
      setTemplatesOpen(false)
    }
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") setTemplatesOpen(false)
    }
    document.addEventListener("pointerdown", onPointerDown)
    document.addEventListener("keydown", onKeyDown)
    return () => {
      document.removeEventListener("pointerdown", onPointerDown)
      document.removeEventListener("keydown", onKeyDown)
    }
  }, [templatesOpen])

  const streamAssistantReply = async (
    prompt: string,
    options?: {
      document?: File | null
      silent?: boolean
      attachment?: ChatAttachment
      questionId?: string
      displayContent?: string
      headerTitle?: string
      headerDescription?: string
      canvasAnchorId?: string
      canvasRequestId?: string
      canvasBranchFromId?: string
      trackerQuestionId?: string
      /** Stamp the reply so Completed / Needs Review / Skip appear after it finishes. */
      offerEvaluation?: boolean
      explainSource?: "chat" | "desk"
      modeOverride?: StudyMode
      /** Reuse an aborted AI row instead of appending a new user+AI pair. */
      resumeMessageId?: string
      /** Always spawn a fresh board-question card even in Socratic mode —
       * for explicit "new problem" entry points (task tracker JIT start,
       * document summarize/extract) that must never inherit whatever
       * question is currently active in an ongoing Socratic dialogue. */
      forceNewQuestion?: boolean
    },
  ) => {
    const silent = options?.silent ?? false
    const canvasBranchFromId = options?.canvasBranchFromId
    const explainSource = options?.explainSource
    const fileToSend = options?.document ?? null
    if ((!prompt.trim() && !fileToSend) || (isSending && !options?.canvasAnchorId)) return

    const publicQuery = (options?.displayContent ?? prompt).trim()
    if (isDraft && !silent) {
      commitSessionTitle(publicQuery, fileToSend?.name)
    }
    const fallbackHeaderTitle = activeQuestion
      ? `${activeQuestion.chapterTitle}: ${activeQuestion.question.label}`
      : userFacingProblemText(activeProblem?.title) ||
        activeProblem?.title ||
        t("chat.mathCalculation")
    const cardHeaderTitle = options?.headerTitle || fallbackHeaderTitle
    const cardHeaderDescription =
      options?.headerDescription ||
      userFacingProblemText(activeProblem?.description) ||
      activeProblem?.description
    const stampQuestionId = silent
      ? undefined
      : (options?.questionId ?? activeQuestionId ?? undefined)
    const resumeMessageId = options?.resumeMessageId
    const aiMessageId = resumeMessageId ?? crypto.randomUUID()
    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: publicQuery,
      attachment: options?.attachment,
      questionId: stampQuestionId,
    }
    const aiDraft: ChatMessage = {
      id: aiMessageId,
      role: "ai",
      content: "",
      questionId: stampQuestionId,
      retryPrompt: prompt,
      retrySilent: silent,
      retryDisplayContent: options?.displayContent,
      retryCanvasBranchFromId: canvasBranchFromId,
      retryExplainSource: explainSource,
      offerEvaluation: options?.offerEvaluation,
    }

    const effectiveMode = options?.modeOverride ?? studyMode.id
    const isSocraticTurn = !options?.forceNewQuestion && effectiveMode === "socratic"
    let turnAnchorId = options?.canvasAnchorId
    if (!silent && !turnAnchorId && isSocraticTurn) {
      const activeId = getActiveQuestionId()
      if (socraticAnchorRef.current && activeId === socraticAnchorRef.current) {
        turnAnchorId = socraticAnchorRef.current
      }
    }
    const turnRequestId = options?.canvasRequestId || crypto.randomUUID()
    let streamedChat = ""

    if (resumeMessageId) {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === resumeMessageId
            ? { ...aiDraft, offerEvaluation: aiDraft.offerEvaluation || m.offerEvaluation }
            : m,
        ),
      )
    } else {
      setMessages((prev) => [
        ...prev,
        ...(silent ? [] : [userMessage]),
        aiDraft,
      ])
    }
    cancelHighlightDismiss()
    setHighlightAsk(null)
    setIsSending(true)
    setPipelineStatus(
      fileToSend && /pdf/i.test(fileToSend.type || fileToSend.name)
        ? "Indexing document (large textbooks may take up to 1 minute)..."
        : "",
    )
    streamingAiIdRef.current = aiMessageId
    window.getSelection()?.removeAllRanges()
    clearMathSelectionPaint()
    const turnDeskIds: string[] = []
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller
    pendingSolveRef.current = { requestId: turnRequestId, shapeId: turnAnchorId }
    let turnStatusClosed = false
    const closeTurnStatus = () => {
      turnStatusClosed = true
      setPipelineStatus("")
    }

    try {
      const accessToken = authSession?.access_token
      const sessionDocumentPath = getSession(threadId)?.documentPath || ""
      const rememberedPath = documentPathRef.current || sessionDocumentPath
      if (rememberedPath && rememberedPath !== documentPathRef.current) {
        documentPathRef.current = rememberedPath
        writeStoredDocumentPath(threadId, rememberedPath)
      }
      const isPdf = Boolean(fileToSend && /pdf/i.test(fileToSend.type || fileToSend.name))
      const savedDocumentPath = isPdf ? undefined : rememberedPath || undefined
      if (user && accessToken && isPdf && fileToSend) {
        void uploadSessionPdf({
          userId: user.id,
          threadId,
          file: fileToSend,
          accessToken,
        }).then((uploaded) => {
          if (!uploaded) return
          documentPathRef.current = uploaded
          writeStoredDocumentPath(threadId, uploaded)
          patchSession(threadId, { documentPath: uploaded })
        })
      }
      await streamChatCompletion({
        message: prompt,
        threadId,
        mode: effectiveMode as StudyMode,
        uiLanguage: locale,
        document: fileToSend,
        canvasAnchorId: turnAnchorId,
        canvasRequestId: turnRequestId,
        canvasBranchFromId,
        explainSource,
        userId: user?.id,
        accessToken,
        documentPath: savedDocumentPath,
        signal: controller.signal,
        onStatus: (status) => {
          if (turnStatusClosed) return
          if (PIPELINE_STATUSES.has(status)) setPipelineStatus(status)
        },
        onChatChunk: (chunk) => {
          if (explainSource === "desk") return
          streamedChat += chunk
          if (isSocraticTurn && /result found:/i.test(streamedChat)) {
            socraticAnchorRef.current = null
          }
          setMessages((prev) =>
            prev.map((m) =>
              m.id === aiMessageId && !m.aborted
                ? { ...m, content: m.content + chunk }
                : m,
            ),
          )
        },
        onCanvasOp: (raw, meta) => {
          if (explainSource === "chat") return
          if (!isCanvasOp(raw)) return
          if (silent && raw.op === "question") return
          // Follow-up Socratic turns already have a hydrated parent. First
          // turn still applies extract `question` ops so a pending/"Solve
          // the problem" placeholder can receive the PDF stem.
          if (isSocraticTurn && turnAnchorId && raw.op === "question" && !raw.freshCard) return
          if (raw.op === "question" && raw.freshCard) turnAnchorId = undefined
          if (isSocraticTurn && raw.op === "result") socraticAnchorRef.current = null
          applyCanvasOp(raw, {
            canvasAnchorId: meta?.canvasAnchorId || canvasBranchFromId || turnAnchorId,
            requestId: turnRequestId,
            trackerQuestionId: options?.trackerQuestionId,
            cardTitle: options?.trackerQuestionId ? cardHeaderTitle : undefined,
          })
          if (isSocraticTurn && raw.op === "question") {
            const anchored = getActiveQuestionId()
            if (anchored) socraticAnchorRef.current = anchored
          }
        },
        onDeskUpdate: (update, meta) => {
          if (explainSource === "chat" || canvasBranchFromId) return
          const id = pushDeskUpdate(
            {
              ...update,
              headerTitle: cardHeaderTitle,
              headerDescription: cardHeaderDescription,
            },
            {
              canvasAnchorId: meta?.canvasAnchorId || turnAnchorId,
              requestId: turnRequestId,
              trackerQuestionId: options?.trackerQuestionId,
            },
          )
          if (id) turnDeskIds.push(id)
          // Silent follow-ups should not rewrite the Desk header from the
          // synthesized prompt. Backend `active_problem_update` is also
          // ignored here so a Why?/How? click doesn't steal the active problem.
          // Never feed the internal `[TASK START]` routing prompt into titles.
          if (!silent && isDeskUpdate(update)) {
            setActiveProblem(buildActiveProblemFromDeskUpdate(update, publicQuery, fileToSend?.name))
          }
        },
        onActiveProblemUpdate: (problem) => {
          if (silent) return
          if (!isActiveProblemUpdate(problem)) return
          const title = userFacingProblemText(problem.title) || problem.title.trim()
          const description = userFacingProblemText(
            typeof problem.description === "string" ? problem.description : "",
          )
          setActiveProblem({
            title,
            description,
            source: typeof problem.source === "string" ? problem.source : undefined,
            topic: typeof problem.topic === "string" && problem.topic.trim() ? problem.topic.trim() : undefined,
          })
          stampHeadersOnItems(turnDeskIds, title, description)
        },
        onDone: () => {
          closeTurnStatus()
          completeSolve(turnRequestId, turnAnchorId || getActiveQuestionId() || undefined)
          setIsSending(false)
        },
      })
      if (explainSource === "desk" && !controller.signal.aborted) {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === aiMessageId && !m.aborted
              ? { ...m, content: t("chat.explanationAdded") }
              : m,
          ),
        )
      }
    } catch (error) {
      const aborted = controller.signal.aborted || isAbortError(error)
      if (aborted) {
        markMessageAborted(aiMessageId)
        return
      }
      console.error("[ChatPane] Failed to stream response from backend:", error)
      setMessages((prev) =>
        prev.map((m) =>
          m.id === aiMessageId && !m.content
            ? {
                ...m,
                content: t("chat.backendError"),
              }
            : m,
        ),
      )
    } finally {
      if (controller.signal.aborted) {
        markMessageAborted(aiMessageId)
      }
      completeSolve(turnRequestId, turnAnchorId || getActiveQuestionId() || undefined)
      if (pendingSolveRef.current?.requestId === turnRequestId) {
        pendingSolveRef.current = null
      }
      if (streamingAiIdRef.current === aiMessageId) streamingAiIdRef.current = null
      if (abortRef.current === controller) abortRef.current = null
      closeTurnStatus()
      if (options?.offerEvaluation && !controller.signal.aborted) {
        setMessages((prev) =>
          prev.map((message) =>
            message.id === aiMessageId ? { ...message, offerEvaluation: true } : message,
          ),
        )
      }
      setIsSending(false)
    }
  }

  const streamAssistantReplyRef = useRef(streamAssistantReply)
  streamAssistantReplyRef.current = streamAssistantReply

  useEffect(() => {
    registerSolveHandler((prompt, shapeId, requestId, mode) => {
      void streamAssistantReplyRef.current(prompt, {
        silent: true,
        canvasAnchorId: shapeId,
        canvasRequestId: requestId,
        modeOverride: "detailed",
        displayContent: prompt,
      })
      void mode
    })
    return () => registerSolveHandler(null)
  }, [registerSolveHandler])

  const stopGeneration = () => {
    const streamingId = streamingAiIdRef.current
    const pending = pendingSolveRef.current
    abortRef.current?.abort()
    abortRef.current = null
    setIsSending(false)
    if (pending) completeSolve(pending.requestId, pending.shapeId)
    if (streamingId) markMessageAborted(streamingId)
    window.setTimeout(() => inputRef.current?.focus(), 0)
  }

  const redoAborted = (message: ChatMessage) => {
    if (isSendingRef.current || !message.retryPrompt?.trim()) return
    void streamAssistantReply(message.retryPrompt, {
      silent: message.retrySilent,
      displayContent: message.retryDisplayContent,
      questionId: message.questionId,
      resumeMessageId: message.id,
      canvasBranchFromId: message.retryCanvasBranchFromId,
      canvasAnchorId: message.retryCanvasBranchFromId,
      explainSource: message.retryExplainSource,
    })
  }

  const startQuestionJit = useCallback(
    async (questionId: string, displayOverride?: string, detailed?: boolean) => {
      if (isSendingRef.current) return
      let chapterTitle = ""
      let number = 0
      let questionLabel = ""
      for (const chapter of chapters) {
        const question = chapter.questions.find((q) => q.id === questionId)
        if (question) {
          chapterTitle = chapter.title
          number = questionNumber(question)
          questionLabel = question.label
          break
        }
      }
      if (!number || !chapterTitle) return

      if (detailed) {
        setAutoReviewed(questionId, false)
        if (reviewHintHold.current === questionId) reviewHintHold.current = null
      }
      startQuestion(questionId)
      const fileToSend = pendingFileRef.current || attachedFileRef.current
      if (fileToSend) {
        pendingFileRef.current = null
        attachedFileRef.current = null
        setAttachedFile(null)
        setPendingFileBound(false)
      }
      const az = /[əƏıİöÖüÜğĞşŞ]|sual|section/i.test(chapterTitle + (displayOverride || ""))
      const taskPrompt = buildTaskStartPrompt(chapterTitle, number)
      const prompt = detailed
        ? `${taskPrompt} User requested a re-evaluation. Provide a highly detailed, expanded, step-by-step Socratic breakdown. Do not just output the standard compact solution.`
        : taskPrompt
      await streamAssistantReplyRef.current(prompt, {
        document: fileToSend,
        attachment: fileToSend ? toChatAttachment(fileToSend) : undefined,
        questionId,
        displayContent: displayOverride || buildTaskStartDisplay(chapterTitle, number, az),
        headerTitle: `${chapterTitle}: ${questionLabel || `Q${number}`}`,
        forceNewQuestion: true,
        trackerQuestionId: questionId,
      })
    },
    [chapters, setAutoReviewed, startQuestion],
  )

  onChatActionRef.current = (href: string, messageId?: string) => {
    const action = parseChatAction(href)
    if (!action || isSendingRef.current) return
    if (messageId) {
      setMessages((prev) => prev.filter((m) => m.id !== messageId))
    }
    const currentChapters = chaptersRef.current
    if (action.kind === "start-id") {
      const match = currentChapters
        .flatMap((chapter) => chapter.questions)
        .find((question) => question.id === action.questionId)
      if (match) void startQuestionJit(match.id)
      return
    }
    if (action.kind === "start") {
      const match = questionWithNumber(currentChapters, action.number, activeQuestionIdRef.current)
      if (match) void startQuestionJit(match.id)
      return
    }
    const nextId = nextUnansweredId(currentChapters, activeQuestionIdRef.current)
    const target = currentChapters
      .flatMap((chapter) => chapter.questions)
      .find((question) => question.id === nextId && question.status === "unanswered")
    if (target) void startQuestionJit(target.id)
  }

  useEffect(() => {
    setOnStartQuestion((id, detailed) => {
      void startQuestionJit(id, undefined, detailed)
    })
    return () => setOnStartQuestion(null)
  }, [setOnStartQuestion, startQuestionJit])

  useEffect(() => {
    setOnQuestionStatus((questionId, status) => {
      if (status !== "review") return
      const question = chaptersRef.current
        .flatMap((chapter) => chapter.questions)
        .find((item) => item.id === questionId)
      if (!question?.autoReviewed) return
      setMessages((prev) => appendReviewHint(prev, questionId))
    })
    return () => setOnQuestionStatus(null)
  }, [appendReviewHint, setOnQuestionStatus])

  useEffect(() => {
    const questions = chapters.flatMap((chapter) => chapter.questions)
    const chapter = chapters.find((item) => item.questions.some((q) => q.id === activeQuestionId))
    const finished = Boolean(chapter && chapter.questions.every((q) => q.status !== "unanswered"))
    const nextReview = finished
      ? chapter?.questions.find((q) => q.status === "review" && !q.autoReviewed)
      : undefined
    const chipId = nextReview ? `review-chip:${nextReview.id}` : ""
    setMessages((prev) => {
      const staleHintIds = new Set(
        prev
          .filter((message) => message.id.startsWith("review-hint:"))
          .filter((message) => {
            const questionId = message.id.slice("review-hint:".length)
            const question = questions.find((item) => item.id === questionId)
            return !question || question.status !== "review"
          })
          .map((message) => message.id),
      )
      const withoutStale = staleHintIds.size
        ? prev.filter((message) => !staleHintIds.has(message.id))
        : prev
      if (!chipId || !nextReview) {
        const withoutChips = withoutStale.filter((message) => !message.id.startsWith("review-chip:"))
        return withoutChips.length === prev.length ? prev : withoutChips
      }
      const chips = withoutStale.filter((message) => message.id.startsWith("review-chip:"))
      if (chips.length === 1 && chips[0]?.id === chipId && staleHintIds.size === 0) return prev
      return [
        ...withoutStale.filter((message) => !message.id.startsWith("review-chip:")),
        {
          id: chipId,
          role: "ai" as const,
          content: "",
          questionId: nextReview.id,
          suggestedActions: [
            {
              id: `review-action:${nextReview.id}`,
              label: t("chat.reviewChip", { label: nextReview.label }),
              questionId: nextReview.id,
              kind: "review" as const,
            },
          ],
        },
      ]
    })
  }, [chapters, activeQuestionId, t])

  const send = async () => {
    const text = input.trim()
    const composerFile = attachedFile
    if (isSending) return

    // Empty composer + a newly queued file: local intent chips, no API call.
    if (!text && composerFile) {
      pendingFileRef.current = composerFile
      setPendingFileBound(true)
      setAttachedFile(null)
      setMessages((prev) => [
        ...prev.filter((m) => m.intent !== "file-prompt"),
        {
          id: crypto.randomUUID(),
          role: "user",
          content: "",
          attachment: toChatAttachment(composerFile),
        },
        {
          id: crypto.randomUUID(),
          role: "ai",
          content: t("chat.filePrompt"),
          intent: "file-prompt",
        },
      ])
      return
    }

    if (!text) return

    const parsed = parseTaskAssignment(text)
    if (parsed) {
      const fileToKeep = composerFile ?? pendingFileRef.current
      const documentName =
        fileToKeep && /pdf/i.test(fileToKeep.type || fileToKeep.name) ? fileToKeep.name : ""
      const firstNewId = assignModule({ ...parsed, documentName })
      if (fileToKeep) {
        pendingFileRef.current = fileToKeep
        setPendingFileBound(true)
      }
      setInput("")
      setAttachedFile(null)
      setMessages((prev) => [
        ...prev.filter((m) => m.intent !== "file-prompt"),
        {
          id: crypto.randomUUID(),
          role: "user",
          content: text,
          attachment: fileToKeep ? toChatAttachment(fileToKeep) : undefined,
        },
        {
          id: crypto.randomUUID(),
          role: "ai",
          content: buildAssignmentAck(parsed, text, firstNewId),
        },
      ])
      if (isDraft) commitSessionTitle(text, fileToKeep?.name)
      return
    }

    const confirm = parseStartConfirm(text)
    if (confirm) {
      const all = chapters.flatMap((c) => c.questions.map((q) => ({ chapter: c, question: q })))
      const numbered = confirm.kind === "number"
        ? questionWithNumber(chapters, confirm.number, activeQuestionId)
        : undefined
      const match =
        confirm.kind === "number"
          ? numbered
            ? { question: numbered }
            : undefined
          : all.find((item) => item.question.status === "unanswered") ?? all[0]
      if (match) {
        setInput("")
        setAttachedFile(null)
        await startQuestionJit(match.question.id, text)
        return
      }
    }

    const fileToSend = composerFile ?? pendingFileRef.current
    setInput("")
    setAttachedFile(null)
    pendingFileRef.current = null
    setPendingFileBound(false)
    setMessages((prev) => prev.filter((m) => m.intent !== "file-prompt"))
    const questionId = activeQuestionId ?? undefined
    const question = questionId
      ? chaptersRef.current
          .flatMap((chapter) => chapter.questions)
          .find((item) => item.id === questionId)
      : undefined
    const stem = question
      ? (question.stem || readTrackerQuestionPrompt(question.id)).trim().slice(0, 4000)
      : ""
    const prompt = stem
      ? `[Context: Question ${question?.label ?? "Q"} is: "${stem}"]\n\n${text}`
      : text
    await streamAssistantReply(prompt, {
      document: fileToSend,
      attachment: composerFile ? toChatAttachment(composerFile) : undefined,
      displayContent: text,
      questionId,
    })
  }

  const handleFileIntent = (choice: "summarize" | "formulas" | "other" | "tracker") => {
    if (isSending) return
    if (choice === "tracker") {
      const template = PROMPT_TEMPLATES.find((t) => t.id === "section")
      if (template) insertPromptTemplate(template)
      return
    }
    setMessages((prev) => prev.filter((m) => m.intent !== "file-prompt"))
    if (choice === "other") {
      inputRef.current?.focus()
      return
    }
    const file = pendingFileRef.current
    pendingFileRef.current = null
    setPendingFileBound(false)
    const prompt =
      choice === "summarize"
        ? t("chat.summarizeDoc")
        : t("chat.extractDoc")
    void streamAssistantReply(prompt, { document: file, forceNewQuestion: true })
  }

  const insertPromptTemplate = (template: PromptTemplate) => {
    setTemplatesOpen(false)
    const text = t(`templates.${template.id}.text`)
    setInput(text)
    const token = template.selectToken
    const start = token ? text.indexOf(token) : -1
    if (start >= 0 && token) {
      setPendingSelect({ start, end: start + token.length })
    } else {
      setPendingSelect({ start: text.length, end: text.length })
    }
  }

  const askAboutHighlight = () => {
    if (!highlightAsk || isSending) return
    const quoted = highlightAsk.text
    const shapeId = highlightAsk.shapeId
    cancelHighlightDismiss()
    clearGraceTimer()
    clearTipTimer()
    clearMathSelectionPaint()
    setExplainTip(null)
    setHighlightAsk(null)
    const explainSource = shapeId ? "desk" : "chat"
    void streamAssistantReply(buildHighlightAskPrompt(quoted, explainSource), {
      silent: true,
      canvasAnchorId: shapeId,
      canvasBranchFromId: shapeId,
      explainSource,
    })
  }

  const saveHighlightFormula = () => {
    if (!highlightAsk?.formulaTex) return
    const formulaTex = highlightAsk.formulaTex
    const formulaId = saveFormula(formulaTex, "Identifying formula...")
    cancelHighlightDismiss()
    clearGraceTimer()
    clearTipTimer()
    clearMathSelectionPaint()
    setExplainTip(null)
    setHighlightAsk(null)
    void nameSavedFormula({
      formulaId,
      formula: formulaTex,
      userId: user?.id,
      accessToken: authSession?.access_token,
    }).then((name) => {
      if (name) renameFormula(formulaId, name)
    })
  }

  const handleSaveToMemory = (message: ChatMessage) => {
    if (saved[message.id]) return

    if (message.save === "formula") {
      saveFormula(message.formula ?? message.content, message.content)
    } else if (message.save === "graph") {
      const latestChart = [...deskItems].reverse().find((item) => item.type === "chart")
      if (latestChart) {
        saveGraph(latestChart, message.content)
      }
    }
    setSaved((s) => ({ ...s, [message.id]: true }))
  }

  const handleEvaluate = (questionId: string, status: QuestionStatus) => {
    if (status === "review" && reviewHintHold.current === questionId) {
      reviewHintHold.current = null
    }
    setStatus(questionId, status)
    startQuestion(questionId)
    if (status === "review") {
      const ai = [...messages]
        .reverse()
        .find((m) => m.role === "ai" && m.questionId === questionId && m.content)
      const chapter = chapters.find((c) => c.questions.some((q) => q.id === questionId))
      const topic =
        chapter?.title ||
        activeQuestion?.chapterTitle ||
        activeProblem?.title ||
        t("chat.studySession")
      const excerpt = (ai?.content ?? "").slice(0, 2000)
      void recordStruggle({
        thread_id: threadId,
        question_id: questionId,
        topic,
        excerpt,
        formula: ai?.formula,
      }).catch((err) => console.error("[ChatPane] Struggle memory failed:", err))
      saveStruggle(topic, excerpt, questionId)
    }
    const nextId = nextUnansweredId(chapters, questionId)
    const nextQuestion =
      nextId && nextId !== questionId
        ? chapters.flatMap((chapter) => chapter.questions).find((q) => q.id === nextId)
        : undefined
    const chapter = chapters.find((c) => c.questions.some((q) => q.id === questionId))
    const az = looksAzerbaijani(
      `${chapter?.title ?? ""} ${messages.map((m) => m.content).join(" ")}`,
    )
    const reviewedAgain =
      status === "review" &&
      chaptersRef.current
        .flatMap((item) => item.questions)
        .some((question) => question.id === questionId && question.autoReviewed)
    setMessages((prev) => {
      const withPrompt = [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: "ai" as const,
          content: buildNextQuestionPrompt(
            az,
            nextQuestion ? questionNumber(nextQuestion) : undefined,
          ),
        },
      ]
      return reviewedAgain ? appendReviewHint(withPrompt, questionId) : withPrompt
    })
  }

  return (
    <aside
      aria-label={t("chat.aria")}
      className={cn(
        "flex h-full shrink-0 flex-col border-l border-border bg-card/40 origin-bottom-right transition-[transform,opacity] duration-300 ease-out",
        collapsed
          ? "pointer-events-none fixed right-0 bottom-0 z-30 opacity-0"
          : "relative opacity-100",
      )}
      style={{
        width: chatWidth,
        transform: collapsed ? "scale(0.18) translateY(12px)" : "none",
      }}
    >
      {!collapsed ? (
        <div
          role="separator"
          aria-orientation="vertical"
          aria-label={t("chat.resize")}
          aria-valuenow={chatWidth}
          className="absolute top-0 left-0 z-20 h-full w-1.5 -translate-x-1/2 cursor-col-resize touch-none"
          onPointerDown={(event) => {
            event.preventDefault()
            event.currentTarget.setPointerCapture(event.pointerId)
          }}
          onPointerMove={(event) => {
            if (!event.currentTarget.hasPointerCapture(event.pointerId)) return
            onChatWidthChange(window.innerWidth - event.clientX)
          }}
          onPointerUp={(event) => {
            if (!event.currentTarget.hasPointerCapture(event.pointerId)) return
            event.currentTarget.releasePointerCapture(event.pointerId)
            onChatWidthCommit(window.innerWidth - event.clientX)
          }}
        />
      ) : null}
      {/* Header */}
      <div className="flex items-center gap-2.5 border-b border-border px-4 py-3">
        <TutorAvatar className="size-11" />
        <div className="min-w-0">
          <h2 className="truncate font-display text-sm font-semibold">Mr. Locky</h2>
        </div>
        <ModeToggle disabled={isSending} />
        <button
          type="button"
          onClick={onToggle}
          aria-label={t("chat.collapse")}
          title={t("chat.collapse")}
          className="grid size-8 shrink-0 place-items-center rounded-lg text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
        >
          <ChevronsRight className="size-4" aria-hidden="true" />
        </button>
      </div>

      {/* Messages */}
      <div ref={scrollRef} className="scroll-slim flex-1 space-y-4 overflow-y-auto p-4">
        {messages.map((m, index) => (
          <MessageBubble
            key={m.id}
            message={m}
            saved={!!saved[m.id]}
            onSave={() => handleSaveToMemory(m)}
            onAiMouseUp={captureHighlightAsk}
            onFileIntent={handleFileIntent}
            intentDisabled={isSending}
            showTrackerIntent={
              m.intent === "file-prompt" &&
              messages[index - 1]?.attachment?.kind === "pdf" &&
              (messages[index - 1]?.attachment?.size ?? 0) > LARGE_PDF_BYTES
            }
            questionStatus={
              m.questionId
                ? chapters
                    .flatMap((c) => c.questions)
                    .find((q) => q.id === m.questionId)?.status
                : undefined
            }
            onEvaluate={
              m.questionId
                ? (status) => handleEvaluate(m.questionId as string, status)
                : undefined
            }
            redoDisabled={isSending}
            statusText={
              isSending && m.role === "ai" && !m.content && m.id === streamingAiIdRef.current
                ? pipelineStatus
                : ""
            }
            onRedo={() => redoAborted(m)}
            generating={isSending && m.id === streamingAiIdRef.current}
            suggestedDisabled={isSending}
            onSuggestedAction={(action) => {
              if (action.kind === "review" && action.questionId) {
                const questionId = action.questionId
                reviewHintHold.current = questionId
                setAutoReviewed(questionId, true)
                setMessages((prev) => prev.filter((message) => message.id !== `review-chip:${questionId}`))
                focusTrackerQuestion(questionId)
                const question = chaptersRef.current
                  .flatMap((chapter) => chapter.questions)
                  .find((item) => item.id === questionId)
                const label = question?.label ?? "Q"
                const visible = t("chat.reviewPrompt", { label })
                const stem = (question?.stem || readTrackerQuestionPrompt(questionId)).trim().slice(0, 4000)
                const prompt = stem
                  ? `[Context: Question ${label} is: "${stem}"]\n\n${visible}`
                  : visible
                void streamAssistantReply(prompt, {
                  explainSource: "chat",
                  displayContent: visible,
                  questionId,
                  offerEvaluation: true,
                })
                return
              }
              if (action.questionId) {
                void startQuestionJit(action.questionId)
                return
              }
              if (action.label.trim()) {
                void streamAssistantReply(action.label)
              }
            }}
            markdownComponents={actionMarkdown}
          />
        ))}
      </div>

      {highlightAsk ? (
        <HighlightAskToolbar
          highlight={highlightAsk}
          disabled={isSending}
          toolbarRef={highlightToolbarRef}
          onExplain={askAboutHighlight}
          onSave={saveHighlightFormula}
          onChipPointerDown={cancelHighlightDismiss}
        />
      ) : null}
      {explainTip && !highlightAsk ? <ExplainTip tip={explainTip} tipRef={explainTipRef} /> : null}

      {/* Input bar */}
      <div className="border-t border-border p-3">
        {attachedFile && isLargePdf(attachedFile) && (
          <div className="mb-2 rounded-xl border border-border bg-secondary/50 px-3 py-2">
            <p className="text-xs text-muted-foreground">{t("chat.addTrackerPrompt")}</p>
            <button
              type="button"
              disabled={isSending}
              onClick={() => {
                const template = PROMPT_TEMPLATES.find((t) => t.id === "section")
                if (template) insertPromptTemplate(template)
              }}
              className="mt-1.5 rounded-lg bg-secondary px-2.5 py-1 text-xs font-semibold text-muted-foreground transition-colors hover:bg-accent hover:text-foreground disabled:cursor-not-allowed disabled:opacity-50"
            >
              {t("chat.addTracker")}
            </button>
          </div>
        )}

        {attachedFile && (
          <div className="mb-2">
            <PendingFileChip
              file={attachedFile}
              disabled={isSending}
              onRemove={() => setAttachedFile(null)}
            />
          </div>
        )}

        <div className="flex items-start gap-2 rounded-2xl border border-border bg-background p-2 focus-within:border-primary/50">
          {/* Document attachment (sent as `document` in the FormData payload) */}
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*,.pdf,.doc,.docx,.ppt,.pptx,application/pdf"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0] ?? null
              if (!file) {
                e.target.value = ""
                return
              }
              pendingFileRef.current = null
              setPendingFileBound(false)
              if (file.type.startsWith("image/")) {
                attachImageFile(file)
              } else {
                setAttachedFile(file)
              }
              e.target.value = ""
            }}
          />
          <div
            ref={templatesMenuRef}
            className="relative flex shrink-0 items-center gap-1 self-start"
          >
            {templatesOpen ? (
              <div
                role="menu"
                aria-label={t("chat.templates")}
                className="absolute bottom-full left-0 z-40 mb-1.5 w-56 origin-bottom-left rounded-xl border border-border bg-popover/95 p-1 shadow-xl backdrop-blur-md"
              >
                {PROMPT_TEMPLATES.map((template) => (
                  <button
                    key={template.id}
                    type="button"
                    role="menuitem"
                    title={t(`templates.${template.id}.text`)}
                    onClick={() => insertPromptTemplate(template)}
                    className="flex w-full rounded-lg px-2.5 py-2 text-left text-xs font-semibold text-foreground transition-colors hover:bg-accent"
                  >
                    {t(`templates.${template.id}.label`)}
                  </button>
                ))}
              </div>
            ) : null}
            <button
              type="button"
              aria-label={t("chat.templates")}
              aria-haspopup="menu"
              aria-expanded={templatesOpen}
              onClick={() => setTemplatesOpen((open) => !open)}
              disabled={isSending}
              className={cn(
                "grid size-9 shrink-0 place-items-center rounded-xl transition-colors disabled:cursor-not-allowed disabled:opacity-40",
                templatesOpen
                  ? "bg-brand-gradient text-white"
                  : "bg-secondary text-muted-foreground hover:bg-accent hover:text-foreground",
              )}
            >
              <Plus className="size-4" aria-hidden="true" />
            </button>
            <button
              type="button"
              aria-label={t("chat.attach")}
              onClick={() => fileInputRef.current?.click()}
              disabled={isSending}
              className={cn(
                "grid size-9 shrink-0 place-items-center rounded-xl transition-colors disabled:cursor-not-allowed disabled:opacity-40",
                attachedFile || pendingFileBound
                  ? "bg-brand-gradient text-white"
                  : "bg-secondary text-muted-foreground hover:bg-accent hover:text-foreground",
              )}
            >
              <Paperclip className="size-4" aria-hidden="true" />
            </button>
          </div>

          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onPaste={(e) => {
              const items = e.clipboardData?.items
              if (!items) return
              for (const item of items) {
                if (!item.type.startsWith("image/")) continue
                const blob = item.getAsFile()
                if (!blob) continue
                e.preventDefault()
                attachImageFile(blob)
                return
              }
            }}
            onKeyDown={(e) => {
              if (
                e.key === "Enter" &&
                !e.shiftKey &&
                !e.nativeEvent.isComposing &&
                e.keyCode !== 229
              ) {
                e.preventDefault()
                send()
              }
            }}
            rows={1}
            disabled={isSending}
            placeholder={t("chat.placeholder")}
            className="block min-h-9 flex-1 resize-none self-start bg-transparent py-2 text-sm leading-5 outline-none placeholder:text-muted-foreground disabled:opacity-60"
          />

          <button
            type="button"
            onClick={isSending ? stopGeneration : send}
            disabled={!isSending && !input.trim() && !attachedFile}
            aria-label={isSending ? t("chat.stop") : t("chat.send")}
            className="bg-brand-gradient grid size-9 shrink-0 self-start place-items-center rounded-xl text-white transition-all hover:-translate-y-px disabled:cursor-not-allowed disabled:opacity-40"
          >
            {isSending ? (
              <Square className="size-3.5 fill-current" aria-hidden="true" />
            ) : (
              <ArrowUp className="size-5" aria-hidden="true" />
            )}
          </button>
        </div>
      </div>
    </aside>
  )
}

function EvalButton({
  status,
  current,
  onClick,
  icon: Icon,
  label,
  className,
}: {
  status: QuestionStatus
  current: QuestionStatus
  onClick: () => void
  icon: typeof CheckCircle2
  label: string
  className?: string
}) {
  const active = status === current
  return (
    <button
      type="button"
      onClick={onClick}
      title={label}
      aria-label={label}
      aria-pressed={active}
      className={cn(
        "flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-xs font-semibold text-white transition-all hover:-translate-y-px",
        className,
        active ? "ring-2 ring-foreground/30" : "opacity-80 hover:opacity-100",
      )}
    >
      <Icon className="size-3.5" aria-hidden="true" />
      <span>{label}</span>
    </button>
  )
}

function SuggestedActionChips({
  actions,
  disabled,
  onPick,
}: {
  actions: SuggestedAction[]
  disabled?: boolean
  onPick?: (action: SuggestedAction) => void
}) {
  const [dismissed, setDismissed] = useState(false)
  if (dismissed || actions.length === 0) return null

  return (
    <div className="mt-1.5 flex flex-wrap gap-1.5">
      {actions.map((action) => (
        <button
          key={action.id}
          type="button"
          disabled={disabled}
          onClick={() => {
            setDismissed(true)
            onPick?.(action)
          }}
          className="rounded-full border border-primary/25 bg-primary/10 px-3 py-1 text-xs font-semibold text-primary transition-colors hover:bg-primary/15 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {action.label}
        </button>
      ))}
    </div>
  )
}

function EvaluationWidget({
  questionStatus,
  onEvaluate,
  offerAgain,
}: {
  questionStatus: QuestionStatus
  onEvaluate: (status: QuestionStatus) => void
  /** A review explanation asks for a new choice even though the question is already red. */
  offerAgain?: boolean
}) {
  const { t } = useLanguage()
  const [isEvaluated, setIsEvaluated] = useState(!offerAgain && questionStatus !== "unanswered")

  if (isEvaluated || (!offerAgain && questionStatus !== "unanswered")) return null

  const pick = (status: QuestionStatus) => {
    onEvaluate(status)
    setIsEvaluated(true)
  }

  return (
    <div className="mt-1.5 flex flex-wrap items-center gap-1">
      <EvalButton
        status="completed"
        current={questionStatus}
        onClick={() => pick("completed")}
        icon={CheckCircle2}
        label={t("chat.completed")}
        className="bg-emerald-500 hover:bg-emerald-600"
      />
      <EvalButton
        status="review"
        current={questionStatus}
        onClick={() => pick("review")}
        icon={XCircle}
        label={t("chat.needsReview")}
        className="bg-red-500 hover:bg-red-600"
      />
      <EvalButton
        status="skipped"
        current={questionStatus}
        onClick={() => pick("skipped")}
        icon={MinusCircle}
        label={t("chat.skip")}
        className="bg-muted-foreground/70 hover:bg-muted-foreground"
      />
    </div>
  )
}

function MessageBubble({
  message,
  saved,
  onSave,
  onAiMouseUp,
  onFileIntent,
  intentDisabled,
  showTrackerIntent,
  questionStatus,
  onEvaluate,
  suggestedDisabled,
  onSuggestedAction,
  markdownComponents: bubbleMarkdown,
  redoDisabled,
  statusText,
  onRedo,
  generating,
}: {
  message: ChatMessage
  saved: boolean
  onSave: () => void
  onAiMouseUp?: (event?: Event) => void
  onFileIntent?: (choice: "summarize" | "formulas" | "other" | "tracker") => void
  intentDisabled?: boolean
  showTrackerIntent?: boolean
  questionStatus?: QuestionStatus
  onEvaluate?: (status: QuestionStatus) => void
  suggestedDisabled?: boolean
  onSuggestedAction?: (action: SuggestedAction) => void
  markdownComponents?: Components
  redoDisabled?: boolean
  statusText?: string
  onRedo?: () => void
  generating?: boolean
}) {
  const { t } = useLanguage()
  const { studyMode } = useStudyMode()
  const showEvaluationWidget = Boolean(
    message.content &&
      !generating &&
      message.questionId &&
      onEvaluate &&
      questionStatus &&
      (message.offerEvaluation ||
        /\bresult found:/i.test(message.content) ||
        studyMode.id === "detailed"),
  )
  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="flex max-w-[85%] flex-col items-end gap-1.5">
          {message.content ? (
            <div className="bg-brand-gradient rounded-2xl rounded-br-md px-3.5 py-2.5 text-sm leading-relaxed whitespace-pre-wrap text-white shadow-sm">
              {message.content}
            </div>
          ) : null}
          {message.attachment ? (
            <AttachmentChip attachment={message.attachment} variant="muted" />
          ) : null}
        </div>
      </div>
    )
  }

  if (message.aborted) {
    return <AbortedNotice disabled={redoDisabled} onRedo={onRedo} />
  }

  if (message.intent === "file-prompt") {
    return (
      <div className="flex gap-2.5">
        <TutorAvatar className="mt-0.5 size-10" />
        <div className="min-w-0 flex-1">
          <div className="rounded-2xl rounded-tl-md border border-border bg-card px-3.5 py-2.5 text-sm leading-relaxed text-card-foreground shadow-sm">
            <p>{message.content}</p>
            <div className="mt-2.5 flex flex-wrap gap-1.5">
              {(
                [
                  ["summarize", t("chat.summarize")],
                  ["formulas", t("chat.extractFormulas")],
                  ...(showTrackerIntent
                    ? ([["tracker", t("chat.addTrackerPrompt")]] as const)
                    : []),
                  ["other", t("chat.other")],
                ] as const
              ).map(([choice, label]) => (
                <button
                  key={choice}
                  type="button"
                  disabled={intentDisabled}
                  onClick={() => onFileIntent?.(choice)}
                  className="rounded-lg bg-secondary px-2.5 py-1 text-xs font-semibold text-muted-foreground transition-colors hover:bg-accent hover:text-foreground disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {label}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="flex gap-2.5">
      <TutorAvatar className="mt-0.5 size-10" />
      <div className="min-w-0 flex-1">
        {message.content || !message.suggestedActions?.length ? (
        <div
          data-ai-message={message.content ? "" : undefined}
          onMouseUp={message.content ? (event) => onAiMouseUp?.(event.nativeEvent) : undefined}
          className="rounded-2xl rounded-tl-md border border-border bg-card px-3.5 py-2.5 text-sm leading-relaxed text-card-foreground shadow-sm selection:bg-primary/25"
        >
          {message.content ? (
            <div className="[&>*:last-child]:mb-0">
              <ChatActionMessageIdContext.Provider value={message.id}>
                <MathMarkdown
                  remarkPlugins={CHAT_REMARK_PLUGINS}
                  rehypePlugins={CHAT_REHYPE_PLUGINS}
                  components={bubbleMarkdown ?? markdownComponents}
                >
                  {texifyDeskMath(message.content)}
                </MathMarkdown>
              </ChatActionMessageIdContext.Provider>
            </div>
          ) : generating ? (
            <span className="flex items-center gap-2 py-0.5" aria-label={statusText || t("chat.thinking")}>
              <span className="flex items-center gap-1" aria-hidden="true">
                <span className="size-1.5 animate-bounce rounded-full bg-muted-foreground/50 [animation-delay:-0.3s]" />
                <span className="size-1.5 animate-bounce rounded-full bg-muted-foreground/50 [animation-delay:-0.15s]" />
                <span className="size-1.5 animate-bounce rounded-full bg-muted-foreground/50" />
              </span>
              <span className="text-sm text-muted-foreground">{statusText || t("chat.thinking")}</span>
            </span>
          ) : null}

          {message.formula && (
            <div className="mt-2.5 rounded-xl bg-secondary/70 px-3 py-2 text-center font-mono text-base font-medium text-foreground">
              {message.formula}
            </div>
          )}
        </div>
        ) : null}

          {message.save && (
          <button
            type="button"
            onClick={onSave}
            disabled={saved}
            className={cn(
              "mt-1.5 inline-flex items-center gap-1.5 rounded-lg border px-2.5 py-1 text-xs font-medium transition-colors",
              saved
                ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400"
                : "border-border bg-background text-muted-foreground hover:border-primary/40 hover:text-foreground",
            )}
          >
            {saved ? (
              <Check className="size-3.5" aria-hidden="true" />
            ) : message.save === "formula" ? (
              <FunctionSquare className="size-3.5" aria-hidden="true" />
            ) : (
              <LineChart className="size-3.5" aria-hidden="true" />
            )}
            {saved
              ? t("chat.saved")
              : message.save === "formula"
                ? t("chat.saveFormula")
                : t("chat.saveGraph")}
          </button>
        )}

        {message.suggestedActions && message.suggestedActions.length > 0 ? (
          <SuggestedActionChips
            actions={message.suggestedActions}
            disabled={suggestedDisabled}
            onPick={onSuggestedAction}
          />
        ) : null}

        {showEvaluationWidget && questionStatus && onEvaluate ? (
          <EvaluationWidget
            questionStatus={questionStatus}
            onEvaluate={onEvaluate}
            offerAgain={message.offerEvaluation}
          />
        ) : null}
      </div>
    </div>
  )
}

function AbortedNotice({
  disabled,
  onRedo,
}: {
  disabled?: boolean
  onRedo?: () => void
}) {
  const { t } = useLanguage()
  return (
    <div className="flex flex-col items-center gap-1.5 py-1">
      <div className="flex w-full items-center gap-3">
        <span className="h-px min-w-0 flex-1 bg-border" />
        <span className="shrink-0 text-[11px] font-medium tracking-wide text-muted-foreground">
          {t("chat.aborted")}
        </span>
        <span className="h-px min-w-0 flex-1 bg-border" />
      </div>
      <button
        type="button"
        disabled={disabled}
        onClick={onRedo}
        aria-label={t("chat.redo")}
        title={t("chat.redo")}
        className="grid size-7 place-items-center rounded-full text-muted-foreground/70 transition-colors hover:bg-accent hover:text-foreground disabled:cursor-not-allowed disabled:opacity-40"
      >
        <RotateCcw className="size-3.5" aria-hidden="true" />
      </button>
    </div>
  )
}

function HighlightAskToolbar({
  highlight,
  disabled,
  toolbarRef,
  onExplain,
  onSave,
  onChipPointerDown,
}: {
  highlight: HighlightAsk
  disabled: boolean
  toolbarRef: RefObject<HTMLDivElement | null>
  onExplain: () => void
  onSave: () => void
  onChipPointerDown: () => void
}) {
  const { t } = useLanguage()
  const [host, setHost] = useState<HTMLElement | null>(null)
  useEffect(() => {
    setHost(document.body)
  }, [])
  if (!host) return null

  return createPortal(
    <div
      ref={toolbarRef}
      role="toolbar"
      aria-label={t("chat.highlightAsk")}
      onPointerDown={(e) => {
        e.stopPropagation()
        onChipPointerDown()
      }}
      onMouseDown={(e) => e.preventDefault()}
      style={{
        position: "fixed",
        left: highlight.x,
        top: highlight.y,
        transform:
          highlight.placement === "above" ? "translate(-50%, -100%)" : "translate(-50%, 0)",
        zIndex: 80,
      }}
      className="flex items-center gap-0.5 rounded-xl border border-border bg-popover/95 p-1 shadow-xl backdrop-blur-md"
    >
      <button
        type="button"
        disabled={disabled}
        onClick={onExplain}
        className="rounded-lg px-2.5 py-1 text-xs font-semibold text-foreground transition-colors hover:bg-accent disabled:cursor-not-allowed disabled:opacity-40"
      >
        {t("chat.explain")}
      </button>
      {highlight.formulaTex ? (
        <button
          type="button"
          disabled={disabled}
          onClick={onSave}
          className="rounded-lg px-2.5 py-1 text-xs font-semibold text-foreground transition-colors hover:bg-accent disabled:cursor-not-allowed disabled:opacity-40"
        >
          {t("chat.saveFormula")}
        </button>
      ) : null}
    </div>,
    host,
  )
}

function ExplainTip({
  tip,
  tipRef,
}: {
  tip: { x: number; y: number; placement: "above" | "below" }
  tipRef: RefObject<HTMLDivElement | null>
}) {
  const { t } = useLanguage()
  const [host, setHost] = useState<HTMLElement | null>(null)
  useEffect(() => {
    setHost(document.body)
  }, [])
  if (!host) return null
  return createPortal(
    <div
      ref={tipRef}
      role="status"
      style={{
        position: "fixed",
        left: tip.x,
        top: tip.y,
        transform: tip.placement === "above" ? "translate(-50%, -100%)" : "translate(-50%, 0)",
        zIndex: 80,
      }}
      className="pointer-events-none max-w-56 rounded-full bg-slate-900/75 px-2.5 py-1 text-[11px] leading-snug text-white/90 shadow-md backdrop-blur-md"
    >
      {t("chat.altExplainTip")}
    </div>,
    host,
  )
}

export function ChatCollapseFab({
  collapsed,
  onExpand,
}: {
  collapsed: boolean
  onExpand: () => void
}) {
  const { t } = useLanguage()
  return (
    <button
      type="button"
      onClick={onExpand}
      aria-label={t("chat.expand")}
      className={cn(
        "fixed right-6 bottom-6 z-40 border-0 bg-transparent p-0 shadow-none outline-none transition-opacity duration-300",
        collapsed ? "opacity-100" : "pointer-events-none opacity-0",
      )}
    >
      <TutorAvatar className="size-[4.4rem] rounded-none border-0 bg-transparent shadow-none" />
    </button>
  )
}
