export type ParsedTaskAssignment = {
  title: string
  questionNumbers: number[]
}

export type StartConfirm =
  | { kind: "first-unanswered" }
  | { kind: "number"; number: number }

function parseNumberList(raw: string): number[] {
  const nums = raw
    .split(/[,;]+|\s+(?:and|&|və)\s+|\s+/)
    .map((part) => part.replace(/[^\d]/g, ""))
    .filter(Boolean)
    .map((part) => Number.parseInt(part, 10))
    .filter((n) => Number.isFinite(n) && n > 0 && n < 500)
  return [...new Set(nums)]
}

export function looksAzerbaijani(text: string): boolean {
  return /[əƏıİöÖüÜğĞşŞ]|əlavə|sual|fəsil|bölmə|başla/i.test(text)
}

export const ACTION_HREF_PREFIX = "#action:"

export type ChatAction =
  | { kind: "next" }
  | { kind: "start"; number: number }
  | { kind: "start-id"; questionId: string }

/** Parses `[label](#action:start_q1)` / `[label](#action:next_question)`. */
export function parseChatAction(href: string | undefined | null): ChatAction | null {
  if (!href) return null
  const raw = href.trim()
  const body = raw.startsWith(ACTION_HREF_PREFIX)
    ? raw.slice(ACTION_HREF_PREFIX.length)
    : raw.startsWith("action:")
      ? raw.slice("action:".length)
      : null
  if (!body) return null
  if (body === "next_question" || body === "next") return { kind: "next" }
  const startId = body.match(/^start_id:(.+)$/i)
  if (startId?.[1]?.trim()) return { kind: "start-id", questionId: startId[1].trim() }
  const start = body.match(/^start_q(\d+)$/i)
  if (!start) return null
  const number = Number.parseInt(start[1], 10)
  if (!Number.isFinite(number) || number <= 0) return null
  return { kind: "start", number }
}

/**
 * Detects a Task Tracker assignment in the student's send text.
 * Handles the Azerbaijani "Section sualları" template, spoken
 * "1 və 2-ci sualları əlavə et", and English "add questions 1 and 2 from chapter 22".
 */
export function parseTaskAssignment(text: string): ParsedTaskAssignment | null {
  const trimmed = text.trim()
  if (!trimmed) return null

  // Both patterns below only accept a NUMERIC chapter/section identifier.
  // The backend's `resolve_chapter_target` (Tier 1) can only resolve a
  // chapter by number — a topic-named capture (e.g. "Section İstilik
  // dinamikası") is guaranteed to dead-end there. Falling through to
  // `return null` here lets the raw text reach the router as a normal
  // message instead of a doomed-to-fail tracker assignment.
  const sectionBu = trimmed.match(
    /(?:section|fəsil|bölmə)\s+(.+?)[-–]\s*d[əe]n\s+bu\s+suallar[ıi]\s+[əe]lav[əe]\s+et:\s*(.+)/i,
  )
  if (sectionBu && /\d/.test(sectionBu[1])) {
    const questionNumbers = parseNumberList(sectionBu[2])
    if (questionNumbers.length > 0) {
      return {
        title: `Section ${sectionBu[1].trim()}`,
        questionNumbers,
      }
    }
  }

  const sectionSpoken = trimmed.match(
    /(?:section|fəsil|bölmə)\s+(.+?)[-–]\s*d[əe]n\s+(.+?)\s+suallar[ıi]?\s+[əe]lav[əe]\s+et/i,
  )
  if (sectionSpoken && /\d/.test(sectionSpoken[1])) {
    const questionNumbers = parseNumberList(sectionSpoken[2])
    if (questionNumbers.length > 0) {
      return {
        title: `Section ${sectionSpoken[1].trim()}`,
        questionNumbers,
      }
    }
  }

  const addFromChapter = trimmed.match(
    /add(?:\s+these)?\s+questions?\s+(.+?)\s+from\s+(?:chapter|section|ch\.?)\s*(\d+)/i,
  )
  if (addFromChapter) {
    const questionNumbers = parseNumberList(addFromChapter[1])
    if (questionNumbers.length > 0) {
      return {
        title: `Chapter ${addFromChapter[2]}`,
        questionNumbers,
      }
    }
  }

  const addAfterChapter = trimmed.match(
    /add(?:\s+these)?\s+questions?\s+from\s+(?:chapter|section|ch\.?)\s*(\d+)\s*[:\-–]?\s*(.+)/i,
  )
  if (addAfterChapter) {
    const questionNumbers = parseNumberList(addAfterChapter[2])
    if (questionNumbers.length > 0) {
      return {
        title: `Chapter ${addAfterChapter[1]}`,
        questionNumbers,
      }
    }
  }

  const chapter = trimmed.match(
    /(?:(serway|cutnell(?:\s*&\s*johnson)?|halliday)\s*[·•,]?\s*)?(?:ch(?:apter)?\.?\s*)(\d+)\D+?(\d+(?:\s*[,;]|\s+(?:and|&|və)\s+|\s+)+\d[\d\s,;veand&]*)/i,
  )
  if (chapter) {
    const questionNumbers = parseNumberList(chapter[3])
    if (questionNumbers.length > 0) {
      const rawBook = chapter[1] || ""
      const book = rawBook
        ? `${rawBook.replace(/\s+/g, " ").replace(/^\w/, (c) => c.toUpperCase())} · `
        : ""
      return {
        title: `${book}Chapter ${chapter[2]}`.trim(),
        questionNumbers,
      }
    }
  }

  return null
}

/** Short confirm to start the next (or a named) tracker question — not an assignment. */
export function parseStartConfirm(text: string): StartConfirm | null {
  const trimmed = text.trim()
  if (!trimmed) return null
  if (
    /^(bəli|beli|h[əe]|ok|okay|yes|yep|start|başla|başlayaq)[.!?…]*$/i.test(trimmed)
  ) {
    return { kind: "first-unanswered" }
  }
  const named = trimmed.match(
    /^(?:sual|question|q)\s*#?\s*(\d+)(?:\s+il[əe]\s+başlayaq)?[.!?…]*$/i,
  )
  if (named) {
    return { kind: "number", number: Number.parseInt(named[1], 10) }
  }
  return null
}

export function buildAssignmentAck(
  parsed: ParsedTaskAssignment,
  sourceText: string,
  questionId?: string | null,
): string {
  const n = parsed.questionNumbers.length
  const first = parsed.questionNumbers[0]
  const href = questionId
    ? `${ACTION_HREF_PREFIX}start_id:${questionId}`
    : `${ACTION_HREF_PREFIX}start_q${first}`
  if (looksAzerbaijani(sourceText)) {
    return `${parsed.title} üçün ${n} sual əlavə olundu. [Sual ${first}](${href}) ilə başlayaq?`
  }
  const noun = n === 1 ? "question" : "questions"
  return `${n} ${noun} added to ${parsed.title}. [Start question ${first}](${href})?`
}

/** Local AI follow-up after Completed / Needs Review / Skip. */
export function buildNextQuestionPrompt(az: boolean, nextNumber?: number): string {
  if (nextNumber == null) {
    return az
      ? "Bu modulun bütün sualları qiymətləndirildi."
      : "All questions in this set have been evaluated."
  }
  const href = `${ACTION_HREF_PREFIX}next_question`
  return az
    ? `Qiymətləndirmə qeydə alındı. [Növbəti suala keçək?](${href})`
    : `Evaluation saved. [Continue to the next question?](${href})`
}

export function buildTaskStartPrompt(chapterTitle: string, questionNumber: number): string {
  return (
    `[TASK START] Extract and solve ONLY question ${questionNumber} from "${chapterTitle}". ` +
    `If [ACTIVE NAVIGATION CONTEXT] already names this same chapter, reuse that Problems page range with locate_marker_in_range(..., section_kind="problems"). ` +
    `If it names a different chapter, or there is no navigation context, call resolve_chapter_target(chapter, section_kind="problems") once for "${chapterTitle}", then locate_marker_in_range(..., section_kind="problems") for this one number. ` +
    `Do not extract or mention sibling questions.`
  )
}

export function buildTaskStartDisplay(chapterTitle: string, questionNumber: number, az: boolean): string {
  return az
    ? `${chapterTitle}, sual ${questionNumber} — çıxar və həll et.`
    : `Extract and solve question ${questionNumber} from ${chapterTitle}.`
}

export function isInternalRoutingText(text: string | undefined | null): boolean {
  if (!text) return false
  return (
    /\[TASK START\]/i.test(text) ||
    /locate_marker_in_range|resolve_chapter_target/i.test(text) ||
    /\[ACTIVE NAVIGATION CONTEXT\]/i.test(text)
  )
}

const SYSTEM_TAG_RE =
  /\[(?:UI LANGUAGE|STUDY MODE INSTRUCTION|TASK START|TASK EXTRACT|TASK_SUMMARY|RAW_PDF_TEXT|VERBATIM_QUESTION_TEXT|TASK_SHORT_DESCRIPTION|OBJECTIVE|ACTIVE NAVIGATION CONTEXT|HANDOFF WARNING)\][^\n[]*/gi

const UNTAGGED_INSTRUCTION_RE =
  /\b(?:DETAILED EXPLANATION MODE|SOCRATIC MODE|Extract-only)\b[^.]*\.?/gi

/** Strip routing tags so they never land on a board-question shape. */
export function stripSystemContextTags(text: string | undefined | null): string {
  if (!text) return ""
  return text
    .replace(SYSTEM_TAG_RE, " ")
    .replace(UNTAGGED_INSTRUCTION_RE, " ")
    .replace(/\s+/g, " ")
    .trim()
}

/** If the same paragraph was concatenated 2–4 times, keep a single copy. */
export function collapseRepeatedPrompt(text: string): string {
  const s = text.trim()
  if (!s) return ""
  const blocks = s.split(/\n{2,}/).map((block) => block.trim()).filter(Boolean)
  if (blocks.length >= 2 && blocks.every((block) => block === blocks[0])) {
    return blocks[0]
  }
  const flat = s.replace(/\s+/g, " ").trim()
  for (const copies of [4, 3, 2]) {
    if (flat.length < copies * 48) continue
    const unitLen = Math.floor(flat.length / copies)
    const unit = flat.slice(0, unitLen).trim()
    if (unit.length < 48) continue
    if (Array.from({ length: copies }, () => unit).join(" ") === flat) {
      const orig = s.split(/\n{2,}/)[0]?.trim()
      return orig && orig.length >= 48 ? orig : unit
    }
  }
  return s
}

const EXTRACT_FILLER_LINE_RE =
  /^(?:i (?:have|just )?(?:the )?(?:extracted |following )?(?:the )?(?:problem|question)(?: text)?\b|here(?:'s| is) (?:the )?(?:extracted )?(?:problem|question)\b|let me extract\b|səhifə\s+\S.{0,80}?sual|page\s+\d+.{0,60}question|tapıldı\b|çıxar(?:dım|ılmış)|aşağıdakı (?:sual|məsələ)|işte (?:soru|problem)|вот (?:извлеч|задача|вопрос)|i extracted\b)/i

const EXTRACT_FILLER_PREFIX_RE =
  /^(?:(?:i have the (?:extracted )?(?:problem|question)(?: text)?[.!,]?\s*)|(?:(?:now )?let me extract(?: it| the (?:problem|question))?(?: (?:exactly|verbatim|faithfully|as[- ]is))?[.!,]?\s*))+/i

const EXTRACT_META_SENTENCE_RE =
  /^(?:i have (?:the )?(?:extracted )?(?:problem|question)(?: text)?(?: here)?|let me extract|i(?:'ll| will) extract|here(?:'s| is) the extract(?:ed)?|extracting (?:the )?(?:problem|question)|now extracting|i found the (?:problem|question)|the (?:problem|question) (?:text )?is as follows|mən (?:problemi|sualı) çıxar|indi çıxararam|izvleku|izvlechen)/i

const GENERIC_QUESTION_PROMPT_RE =
  /^(?:(?:please )?solve(?: the)?(?: this)?(?: the)? problem|həll et(?: bu problemi)?|bunu(?: da)? həll et|solve this|math(?:s)?(?:\s*[&+/]\s*|\s+)?calculation|extract and solve(?: the)?(?: assigned)? questions?(?: \d+(?:\s*[-–]\s*\d+)?)?(?: from .+)?|sual \d+ ilə başla|start question \d+|active workspace(?:\s*[—–-]\s*ready for queries)?|ready for queries|current task)$/i

const BOOK_NAV_MAX_CHARS = 80
const BOOK_NAV_CHAPTER_RE =
  /(?:ch(?:apter)?\.?|f[əe]sil|f[əe]sl(?:in)?|section|b[öo]lm[əe])\s*#?\s*\d+/i
const BOOK_NAV_PROBLEM_RE =
  /(?:problem|question|sual|m[əe]s[əe]l[əe]|q)\s*#?\s*\d+/i

/** Short "Solve Chapter 8 Problem 17" fetch commands — not a physics stem. */
export function isBookNavigationPrompt(text: string | undefined | null): boolean {
  const cleaned = (text ?? "").replace(/\s+/g, " ").trim()
  if (!cleaned || cleaned.length > BOOK_NAV_MAX_CHARS) return false
  return BOOK_NAV_CHAPTER_RE.test(cleaned) && BOOK_NAV_PROBLEM_RE.test(cleaned)
}

function looksLikeProblemStem(sentence: string): boolean {
  const t = sentence.trim()
  if (!t) return false
  if (EXTRACT_META_SENTENCE_RE.test(t) || EXTRACT_FILLER_LINE_RE.test(t)) return false
  if (/^\d+[.)]\s+\S/.test(t)) return true
  if (t.length >= 24) return true
  return t.length >= 8
}

function stripExtractFiller(text: string): string {
  const raw = text.replace(/\r\n/g, "\n").trim()
  if (!raw) return ""
  const withoutObjective = raw
    .split("\n")
    .filter((line) => !/^\s*\[OBJECTIVE\]/i.test(line))
    .join("\n")
    .trim()
  const withoutPrefix = withoutObjective.replace(EXTRACT_FILLER_PREFIX_RE, "").trim()
  const sentences = withoutPrefix
    .split(/(?<!\d\.)(?<=[.!?])\s+|\n+/)
    .map((part) => part.trim())
    .filter(Boolean)
  const start = sentences.findIndex((sentence) => looksLikeProblemStem(sentence))
  if (start >= 0) {
    return sentences.slice(start).join(" ").replace(/\s+/g, " ").trim()
  }
  return withoutPrefix
}

const COACH_SPEAK_RE =
  /^(?:you have read|you already(?: read)?|you(?:'ve| have) (?:read|seen|got)|let(?:'s| us)\b|your turn\b|which quantity\b|guiding question\b|i have extracted\b|assume you(?: have)? read|as you (?:have )?read|now that you(?:'ve| have) read)/i

function splitSentences(text: string): string[] {
  return text
    .split(/(?<!\d\.)(?<=[.!?])\s+|\n+/)
    .map((part) => part.trim())
    .filter(Boolean)
}

function sentenceIsCoachSpeak(sentence: string): boolean {
  const t = sentence.trim()
  if (!t) return false
  return (
    COACH_SPEAK_RE.test(t) ||
    EXTRACT_META_SENTENCE_RE.test(t) ||
    EXTRACT_FILLER_LINE_RE.test(t)
  )
}

const SAME_PROBLEM_THRESHOLD = 0.8

function normalizeStemForMatch(text: string): string {
  const body = isolateProblemStem(text) || text || ""
  return body
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]+/gu, " ")
    .replace(/\s+/g, " ")
    .trim()
}

function diceBigramRatio(left: string, right: string): number {
  if (!left && !right) return 1
  if (!left || !right) return 0
  if (left.length < 2 || right.length < 2) return left === right ? 1 : 0
  const grams = (value: string) => {
    const counts = new Map<string, number>()
    for (let i = 0; i < value.length - 1; i++) {
      const gram = value.slice(i, i + 2)
      counts.set(gram, (counts.get(gram) || 0) + 1)
    }
    return counts
  }
  const a = grams(left)
  const b = grams(right)
  let overlap = 0
  let total = 0
  for (const n of a.values()) total += n
  for (const n of b.values()) total += n
  for (const [gram, n] of a) overlap += Math.min(n, b.get(gram) || 0)
  return total > 0 ? (2 * overlap) / total : 0
}

/** True when two isolated stems are the same problem (spelling / OCR jitter). */
export function stemsAreSameProblem(
  left: string | undefined | null,
  right: string | undefined | null,
  threshold = SAME_PROBLEM_THRESHOLD,
): boolean {
  const a = normalizeStemForMatch(left ?? "")
  const b = normalizeStemForMatch(right ?? "")
  if (!a || !b) return false
  if (a === b) return true
  const tokensA = new Set(a.split(" ").filter(Boolean))
  const tokensB = new Set(b.split(" ").filter(Boolean))
  if (tokensA.size && tokensB.size) {
    let inter = 0
    for (const token of tokensA) if (tokensB.has(token)) inter++
    const union = tokensA.size + tokensB.size - inter
    if (union > 0 && inter / union >= threshold) return true
  }
  const [shorter, longer] = a.length <= b.length ? [a, b] : [b, a]
  if (longer.includes(shorter) && shorter.length / Math.max(longer.length, 1) >= threshold) {
    return true
  }
  return diceBigramRatio(a, b) >= threshold
}

/** Keep the PDF/problem stem; drop RAG `---` coach segments and trailing coaching. */
export function isolateProblemStem(text: string | undefined | null): string {
  const tagged = stripSystemContextTags(text ?? "").replace(/\r\n/g, "\n").trim()
  if (!tagged || isInternalRoutingText(tagged)) return ""
  const kept: string[] = []
  for (const segment of tagged.split(/\s*-{3,}\s*/)) {
    const body = segment.trim()
    if (!body) continue
    const sentences = splitSentences(body)
    if (!sentences.length || sentenceIsCoachSpeak(sentences[0])) continue
    const cut: string[] = []
    for (const sentence of sentences) {
      if (sentenceIsCoachSpeak(sentence)) break
      cut.push(sentence)
    }
    if (cut.length) kept.push(cut.join(" "))
  }
  const joined = kept.join(" ").replace(/\s+/g, " ").trim()
  if (!joined) return ""
  const stripped = stripExtractFiller(joined)
  if (!stripped || isInternalRoutingText(stripped)) return ""
  return collapseRepeatedPrompt(stripped)
}

export function isPlaceholderQuestionPrompt(text: string | undefined | null): boolean {
  const cleaned = isolateProblemStem(text ?? "").trim() || sanitizeCanvasPrompt(text ?? "").trim()
  if (!cleaned) return true
  if (GENERIC_QUESTION_PROMPT_RE.test(cleaned)) return true
  if (isBookNavigationPrompt(cleaned) || isBookNavigationPrompt(text)) return true
  if (/^question\s+\d+$/i.test(cleaned)) return true
  if (/^sual\s+\d+$/i.test(cleaned)) return true
  return !isRenderableQuestionPrompt(cleaned)
}

export function isCoachSpeakPrompt(text: string | undefined | null): boolean {
  const cleaned = sanitizeCanvasPrompt(text ?? "").trim()
  if (!cleaned) return false
  return splitSentences(cleaned).some((sentence) => sentenceIsCoachSpeak(sentence))
}

export function isSubstantialProblemStem(text: string | undefined | null): boolean {
  const cleaned = isolateProblemStem(text ?? "").trim()
  if (!cleaned) return false
  if (GENERIC_QUESTION_PROMPT_RE.test(cleaned)) return false
  if (isBookNavigationPrompt(cleaned) || isBookNavigationPrompt(text)) return false
  if (/^question\s+\d+$/i.test(cleaned)) return false
  if (isCoachSpeakPrompt(cleaned)) return false
  if (/^\d+[.)]\s+\S/.test(cleaned)) return true
  return cleaned.length >= 24
}

/** True only for a real problem stem — not routing, tracker chips, or generic solve prompts. */
export function isRenderableQuestionPrompt(text: string | undefined | null): boolean {
  const cleaned = sanitizeCanvasPrompt(text ?? "").trim()
  if (!cleaned) return false
  if (GENERIC_QUESTION_PROMPT_RE.test(cleaned)) return false
  if (isBookNavigationPrompt(cleaned) || isBookNavigationPrompt(text)) return false
  if (/^question\s+\d+$/i.test(cleaned)) return false
  return cleaned.length >= 8
}

/** Canvas body / header text — never routing tags, never internal prompts. */
export function sanitizeCanvasPrompt(text: string | undefined | null): string {
  const stripped = stripExtractFiller(stripSystemContextTags(text))
  if (!stripped || isInternalRoutingText(stripped)) return ""
  return collapseRepeatedPrompt(stripped)
}

export function userFacingProblemText(text: string | undefined | null): string {
  return sanitizeCanvasPrompt(text)
}

export function buildAssignmentStartChip(
  parsed: ParsedTaskAssignment,
  sourceText: string,
  firstId: string,
): { id: string; label: string; questionId: string } {
  const first = parsed.questionNumbers[0]
  return {
    id: `start-${firstId}`,
    label: looksAzerbaijani(sourceText)
      ? `Sual ${first} ilə başla`
      : `Start question ${first}`,
    questionId: firstId,
  }
}

export const LARGE_PDF_BYTES = 1_000_000

export function isLargePdf(file: File): boolean {
  const type = (file.type || "").toLowerCase()
  const name = file.name.toLowerCase()
  const isPdf = type === "application/pdf" || name.endsWith(".pdf")
  return isPdf && file.size > LARGE_PDF_BYTES
}
