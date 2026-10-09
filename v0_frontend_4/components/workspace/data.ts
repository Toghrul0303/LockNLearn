import type { LucideIcon } from "lucide-react"
import {
  BookMarked,
  FunctionSquare,
  LineChart,
  NotebookPen,
} from "lucide-react"

export type Task = {
  id: string
  label: string
  detail: string
  done: boolean
  flagged: boolean
}

export type QuestionStatus = "unanswered" | "completed" | "review" | "skipped"

export type Question = {
  id: string
  label: string
  /** Textbook problem number (1, 2, 10) — independent of the Q1 badge label. */
  number: number
  status: QuestionStatus
  /** Problem text copied from the board card, so a review still knows the question after reload. */
  stem?: string
  /** The automatic Review chip was already sent for the current explanation. */
  autoReviewed?: boolean
}

export type Chapter = {
  id: string
  title: string
  /** PDF filename this chapter belongs to. Same chapter numbers in different files stay separate. */
  documentName?: string
  questions: Question[]
}

export type MemoryFolder = {
  id: string
  name: string
  count: number
  icon: LucideIcon
  hue: string
}

export type ToolMode = {
  id: string
  label: string
  shortLabel: string
  description: string
}

export type ChatSaveTarget = "formula" | "graph" | null

export type AttachmentKind = "pdf" | "image" | "doc"

export type ChatAttachment = {
  name: string
  mimeType: string
  kind: AttachmentKind
  size?: number
}

export type SuggestedAction = {
  id: string
  label: string
  /** If set, starts this Task Tracker question instead of sending `label`. */
  questionId?: string
  /** Review chips explain in chat. Other chips still start the tracker question. */
  kind?: "review"
}

export type ChatMessage = {
  id: string
  role: "user" | "ai"
  content: string
  save?: ChatSaveTarget
  formula?: string
  attachment?: ChatAttachment
  /** Local-only AI card with Summarize / Extract Formulas / Other. */
  intent?: "file-prompt"
  /** Task Tracker question this turn was answering, if any. */
  questionId?: string
  /** Show Completed / Needs Review / Skip after this reply finishes. */
  offerEvaluation?: boolean
  /** Clickable chips rendered under an AI bubble (e.g. "Sual 1 ilə başla"). */
  suggestedActions?: SuggestedAction[]
  /** Stopped mid-generation — rendered as a divider, not a tutor bubble. */
  aborted?: boolean
  /** Internal prompt used to redo an aborted generation. */
  retryPrompt?: string
  retrySilent?: boolean
  retryDisplayContent?: string
  retryCanvasBranchFromId?: string
  retryExplainSource?: "chat" | "desk"
}

export type PromptTemplateId = "solve" | "formulas" | "section"

export type PromptTemplate = {
  id: PromptTemplateId
  /** If set, the first occurrence is selected so the student can overtype it. */
  selectToken?: string
}

export const PROMPT_TEMPLATES: PromptTemplate[] = [
  { id: "solve" },
  { id: "formulas" },
  { id: "section", selectToken: "[X]" },
]

export const INITIAL_TASKS: Task[] = [
  {
    id: "t1",
    label: "Chapter 22 · Q2, Q4, Q5",
    detail: "Rotational dynamics — torque problems",
    done: true,
    flagged: false,
  },
  {
    id: "t2",
    label: "Chapter 22 · Q8",
    detail: "Moment of inertia of a compound body",
    done: true,
    flagged: true,
  },
  {
    id: "t3",
    label: "Chapter 23 · Q1, Q3",
    detail: "Angular momentum conservation",
    done: true,
    flagged: false,
  },
  {
    id: "t4",
    label: "Lab Report · Section 4",
    detail: "Error propagation write-up",
    done: false,
    flagged: true,
  },
  {
    id: "t5",
    label: "Chapter 24 · Q6, Q7",
    detail: "Simple harmonic motion",
    done: false,
    flagged: false,
  },
  {
    id: "t6",
    label: "Practice Set · Waves",
    detail: "Standing waves & resonance",
    done: false,
    flagged: false,
  },
]

/** Builds a chapter of exactly 10 questions (Q1…Q10). */
export function buildChapter(
  id: string,
  title: string,
  seed: Partial<Record<number, QuestionStatus>> = {},
): Chapter {
  return {
    id,
    title,
    questions: Array.from({ length: 10 }, (_, i) => {
      const n = i + 1
      return {
        id: `${id}-q${n}`,
        label: `Q${n}`,
        number: n,
        status: seed[n] ?? "unanswered",
      }
    }),
  }
}

export const INITIAL_CHAPTERS: Chapter[] = [
  buildChapter("serway-22", "Serway · Chapter 22", {
    1: "completed",
    2: "completed",
    3: "review",
    4: "completed",
    5: "skipped",
  }),
  buildChapter("serway-23", "Serway · Chapter 23", {
    1: "completed",
    2: "review",
  }),
]

export function questionNumber(q: Pick<Question, "label" | "number">): number {
  if (typeof q.number === "number" && q.number > 0) return q.number
  const match = q.label.match(/(\d+)/)
  return match ? Number.parseInt(match[1], 10) : 0
}

export function normalizeModuleTitle(title: string): string {
  return title.trim().toLowerCase()
}

/** ASCII `i` does not fold Ə/İ, so those capitals are written into the pattern. */
const CHAPTER_NUMBER_RE =
  /(?<![A-Za-zƏəÖöİi])(?:chapters?|sections?|f[əƏ]s[iİ]ll[əƏ]r|f[əƏ]s[iİ]l|b[öÖ]lm[əƏ]l[əƏ]r|b[öÖ]lm[əƏ]|ch)\.?\s*(\d+)/i

export function chapterNumberFromTitle(title: string): number | null {
  const match = title.match(CHAPTER_NUMBER_RE)
  if (!match) return null
  const number = Number.parseInt(match[1], 10)
  return Number.isFinite(number) && number > 0 ? number : null
}

function documentsCompatible(a?: string, b?: string): boolean {
  const left = normalizeModuleTitle(a ?? "")
  const right = normalizeModuleTitle(b ?? "")
  if (!left || !right) return true
  return left === right
}

export function sameChapterGroup(a: Chapter, b: Chapter): boolean {
  const left = chapterNumberFromTitle(a.title)
  const right = chapterNumberFromTitle(b.title)
  if (left != null && right != null) {
    return left === right && documentsCompatible(a.documentName, b.documentName)
  }
  if (left != null || right != null) return false
  return (
    normalizeModuleTitle(a.title) === normalizeModuleTitle(b.title) &&
    documentsCompatible(a.documentName, b.documentName)
  )
}

function mergeChapterPair(primary: Chapter, incoming: Chapter): Chapter {
  const documentName = primary.documentName || incoming.documentName
  const have = new Set(primary.questions.map(questionNumber))
  const added = incoming.questions.filter((question) => !have.has(questionNumber(question)))
  if (added.length === 0 && (primary.documentName || "") === (documentName || "")) return primary
  return {
    ...primary,
    documentName,
    questions: [...primary.questions, ...added].sort(
      (a, b) => questionNumber(a) - questionNumber(b),
    ),
  }
}

/** Folds duplicate chapter cards that share a number and a compatible PDF name. */
export function collapseChapters(chapters: Chapter[]): Chapter[] {
  const next: Chapter[] = []
  let changed = false
  for (const chapter of chapters) {
    const index = next.findIndex((existing) => sameChapterGroup(existing, chapter))
    if (index === -1) {
      next.push(chapter)
      continue
    }
    changed = true
    next[index] = mergeChapterPair(next[index], chapter)
  }
  return changed ? next : chapters
}

function questionsFromNumbers(chapterId: string, questionNumbers: number[]): Question[] {
  const unique = [...new Set(questionNumbers.filter((n) => n > 0))].sort((a, b) => a - b)
  return unique.map((n) => ({
    id: `${chapterId}-q${n}`,
    label: `Q${n}`,
    number: n,
    status: "unanswered" as const,
  }))
}

/** Builds a tracker module from an assigned title + question number list. */
export function buildAssignedChapter(
  title: string,
  questionNumbers: number[],
  documentName = "",
): Chapter {
  const slug =
    title
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-|-$/g, "") || "module"
  const id = `${slug}-${Date.now()}`
  return {
    id,
    title,
    documentName: documentName || undefined,
    questions: questionsFromNumbers(id, questionNumbers),
  }
}

/** Appends new question numbers onto an existing section; skips duplicates. */
export function appendQuestionsToChapter(chapter: Chapter, questionNumbers: number[]): Chapter {
  const have = new Set(chapter.questions.map(questionNumber))
  const incoming = [...new Set(questionNumbers.filter((n) => n > 0 && !have.has(n)))].sort(
    (a, b) => a - b,
  )
  if (incoming.length === 0) return chapter
  const added = questionsFromNumbers(chapter.id, incoming)
  return {
    ...chapter,
    questions: [...chapter.questions, ...added].sort(
      (a, b) => questionNumber(a) - questionNumber(b),
    ),
  }
}

export const MEMORY_FOLDERS: MemoryFolder[] = [
  {
    id: "formula",
    name: "FormulaBox",
    count: 24,
    icon: FunctionSquare,
    hue: "var(--brand-red)",
  },
  {
    id: "bookmarks",
    name: "Saved Solutions",
    count: 5,
    icon: BookMarked,
    hue: "var(--brand-red)",
  },
  {
    id: "summaries",
    name: "Summaries",
    count: 8,
    icon: NotebookPen,
    hue: "var(--brand)",
  },
  {
    id: "graphs",
    name: "Graphs",
    count: 11,
    icon: LineChart,
    hue: "var(--brand-purple)",
  },
]

/**
 * IDs must exactly match the backend's supported `mode` values
 * (`detailed` | `socratic`) since `tool.id` is sent
 * directly in the `/submit_stream` FormData payload.
 */
export const TOOL_MODES: ToolMode[] = [
  {
    id: "detailed",
    label: "Detailed Explanation",
    shortLabel: "Detailed",
    description: "Full walkthrough with concepts and steps",
  },
  {
    id: "socratic",
    label: "Socratic Tutor",
    shortLabel: "Socratic",
    description: "Guiding questions and hints — no full solution",
  },
]

export const INITIAL_MESSAGES: ChatMessage[] = [
  {
    id: "m1",
    role: "user",
    content: "How do I find the torque on the rotating disc in Chapter 22, Q4?",
  },
  {
    id: "m2",
    role: "ai",
    content:
      "Torque is the rotational analogue of force. For your disc, use the relationship between torque, moment of inertia, and angular acceleration. Substitute the disc's inertia and you can solve directly.",
    save: "formula",
    formula: "\u03C4 = I\u03B1 = \u00BD M R\u00B2 \u03B1",
  },
  {
    id: "m3",
    role: "user",
    content: "Can you plot how torque changes as angular acceleration increases?",
  },
  {
    id: "m4",
    role: "ai",
    content:
      "Here's the linear relationship — torque scales directly with angular acceleration for a fixed moment of inertia. I've plotted it on the Desk so you can annotate it there.",
    save: "graph",
  },
]

export const NAV_TABS = ["desk"] as const

export type NavTabId = (typeof NAV_TABS)[number]
