import remarkMath from "remark-math"
import rehypeMathjax from "rehype-mathjax/browser"
import type { PluggableList } from "unified"

/**
 * remark-math still parses `$...$` / `$$...$$`.
 *
 * rehype-mathjax/svg cannot run in Next.js client components: mathjax-full/js
 * is a Node CJS build (`__dirname` / `require`) and Turbopack fails to
 * bundle it. The /browser plugin only wraps TeX in `\(...\)` / `\[...\]`.
 * `MathMarkdown` then typesets those markers with mathjax-full's prebundled
 * `es5/tex-svg.js` — still SVG output, still no KaTeX/CSS fractions.
 */
export const mathRemarkPlugins: PluggableList = [remarkMath]

export const mathRehypePlugins: PluggableList = [
  [
    rehypeMathjax,
    {
      tex: {
        inlineMath: [["\\(", "\\)"]],
        displayMath: [["\\[", "\\]"]],
      },
    },
  ],
]

const SCI_E_RE = /(?<![A-Za-z\\])(\d+\.?\d*)[eE]([+-]?\d+)/g
const SUB_IDENT_RE = /(?<![\\$A-Za-z0-9])([A-Za-z])_([A-Za-z][A-Za-z0-9]*)\b/g
const TIMES_TEN_RE = /(?<!\$)(\d+\.?\d*)\s*\\times\s*10\^\{([^{}]+)\}/g
const LEADING_LABEL_RE = /^((?:Sual|Question|Q|Step)\s*\d+\s*:\s*)/i
const PROSE_WORD_RE = /[^\W\d_]{2,}/gu
const HAS_LATEX_COMMAND_RE = /\\[A-Za-z]/
const LATEX_COMMAND_RE = /\\[A-Za-z]+/g
const LATEX_GROUP_RE = /^(?:\{[^{}]*\}|\[[^[\]]*\])/
const LATEX_RIGHT_TIGHT_RE = /[A-Za-z0-9^_+\-*/=(){}[\]\\]/
const LATEX_LEFT_TIGHT_RE = /[A-Za-z0-9)}\]^_=+\-*/]/
const MATHY_TOKEN_RE = /^[0-9+\-*/^_=(){}.,]+$/
const CURRENCY_AMOUNT_RE = /^\d{1,3}(?:,\d{3})*(?:\.\d{2})?$/
const PAREN_DISPLAY_RE = /\\\[([\s\S]*?)\\\]/g
const PAREN_INLINE_RE = /\\\(([\s\S]*?)\\\)/g

type MathSpan = { start: number; end: number; kind: "inline" | "display"; body: string }

function looksLikeProse(text: string): boolean {
  const words = text.match(PROSE_WORD_RE)
  return (words?.length ?? 0) >= 3
}

function pythonEToLatex(text: string): string {
  return text.replace(SCI_E_RE, (_all, coeff: string, exp: string) => {
    const expI = Number.parseInt(exp, 10)
    if (!Number.isFinite(expI)) return _all
    return `${coeff} \\times 10^{${expI}}`
  })
}

function looksLikeMathInterior(interior: string): boolean {
  if (!interior) return false
  const trimmed = interior.trim()
  if (CURRENCY_AMOUNT_RE.test(trimmed)) return false
  if (HAS_LATEX_COMMAND_RE.test(interior)) return true
  if (/[\\^_={}]/.test(interior)) return true
  if (/^\d+(?:\.\d+)?$/.test(trimmed)) return true
  return /\d/.test(interior) && /[+\-*/=]/.test(interior)
}

/**
 * remark-math / Pandoc: an opening `$` cannot be followed by a digit or
 * whitespace, so `$2.0\text{ m/s}^2$` is left as literal text (raw `$`,
 * visible `\text`). Prefix an empty group so the opener is `$` + `{`.
 */
function wrapAsInlineMath(tex: string): string {
  const trimmed = tex.trim()
  const complex = /\\(?:dfrac|tfrac|frac|binom|over)\b|[\^_]/.test(trimmed)
  const body =
    complex && !/^\\displaystyle\b/.test(trimmed) ? `\\displaystyle ${trimmed}` : trimmed
  if (/^[\s\d]/.test(trimmed)) return `$\{}${body}$`
  return `$${body}$`
}

function findDelimitedMath(text: string): MathSpan[] {
  const spans: MathSpan[] = []
  let i = 0
  while (i < text.length) {
    if (text.startsWith("$$", i)) {
      const close = text.indexOf("$$", i + 2)
      if (close === -1) break
      spans.push({ start: i, end: close + 2, kind: "display", body: text.slice(i + 2, close) })
      i = close + 2
      continue
    }
    if (text[i] === "$") {
      const nl = text.indexOf("\n", i + 1)
      const limit = nl === -1 ? text.length : nl
      let j = i + 1
      let accepted = false
      while (j < limit) {
        if (text[j] === "$" && text[j + 1] !== "$") {
          const body = text.slice(i + 1, j)
          const digitOpen = /^\d/.test(body)
          if (digitOpen && !looksLikeMathInterior(body)) {
            j += 1
            continue
          }
          spans.push({ start: i, end: j + 1, kind: "inline", body })
          i = j + 1
          accepted = true
          break
        }
        j += 1
      }
      if (!accepted) i += 1
      continue
    }
    i += 1
  }
  return spans
}

function normalizeParenDelimiters(text: string): string {
  return text
    .replace(PAREN_DISPLAY_RE, (_all, body: string) => `$$${body}$$`)
    .replace(PAREN_INLINE_RE, (_all, body: string) => wrapAsInlineMath(body))
}

function extendLatexRunLeft(text: string, start: number): number {
  let i = start
  while (i > 0 && LATEX_LEFT_TIGHT_RE.test(text[i - 1])) i -= 1
  return i
}

function extendLatexRunRight(text: string, start: number): number {
  let i = start
  while (i < text.length) {
    const ch = text[i]
    const groupMatch = text.slice(i).match(LATEX_GROUP_RE)
    if (groupMatch) {
      i += groupMatch[0].length
      continue
    }
    if (ch === "." || ch === ",") {
      if (/\d/.test(text[i + 1] ?? "")) {
        i += 1
        continue
      }
      break
    }
    if (LATEX_RIGHT_TIGHT_RE.test(ch)) {
      i += 1
      continue
    }
    if (ch === " ") {
      const next = text.slice(i + 1).match(/^(\S+)/)
      if (!next) break
      const nextToken = next[1]
      if (/\\[A-Za-z]/.test(nextToken) || MATHY_TOKEN_RE.test(nextToken)) {
        i += 1
        continue
      }
      break
    }
    break
  }
  return i
}

/**
 * Finds bare `\command...` runs (undelimited LaTeX — the common shape of
 * vision-extracted problem text, which is rarely pre-wrapped in `$...$`) and
 * wraps just that run, leaving surrounding prose and its spaces untouched.
 * `{...}` groups stay intact so `\text{ m/s}` keeps its interior spaces.
 */
export function wrapInlineLatexRuns(text: string): string {
  if (!HAS_LATEX_COMMAND_RE.test(text)) return text
  let result = ""
  let cursor = 0
  const re = new RegExp(LATEX_COMMAND_RE.source, "g")
  let match: RegExpExecArray | null
  while ((match = re.exec(text)) !== null) {
    if (match.index < cursor) continue
    const start = extendLatexRunLeft(text, match.index)
    const end = extendLatexRunRight(text, match.index + match[0].length)
    result += text.slice(cursor, start)
    result += wrapAsInlineMath(text.slice(start, end))
    cursor = end
    re.lastIndex = end
  }
  result += text.slice(cursor)
  return result
}

function texifyPlainSegment(segment: string): string {
  if (!segment) return segment
  let converted = pythonEToLatex(segment)
  converted = converted.replace(TIMES_TEN_RE, (_all, coeff: string, exp: string) => {
    return wrapAsInlineMath(`${coeff} \\times 10^{${exp}}`)
  })
  converted = converted.replace(SUB_IDENT_RE, (_all, base: string, sub: string) => {
    return `$${base}_{${sub}}$`
  })
  if (HAS_LATEX_COMMAND_RE.test(converted)) {
    converted = wrapInlineLatexRuns(converted)
  }
  if (converted.includes("$")) return converted
  if (!/[=]|\\times/.test(converted) || !/\d/.test(converted)) return converted
  if (looksLikeProse(converted)) return converted
  const label = converted.match(LEADING_LABEL_RE)
  if (label) {
    const rest = converted.slice(label[0].length).trim()
    if (rest && looksLikeProse(rest)) return converted
    return rest ? `${label[0]}${wrapAsInlineMath(rest)}` : converted
  }
  const stripped = converted.trim()
  const start = converted.indexOf(stripped)
  return `${converted.slice(0, start)}${wrapAsInlineMath(stripped)}${converted.slice(start + stripped.length)}`
}

function emitMathSpan(span: MathSpan): string {
  const body = pythonEToLatex(span.body)
  if (span.kind === "display") return `$$${body}$$`
  return wrapAsInlineMath(body)
}

/** Idempotent: turns Python-ish desk math (`1.07e4`, bare `Q_h`) into `$...$` TeX. */
export function texifyDeskMath(text: string): string {
  if (!text) return ""
  const normalized = normalizeParenDelimiters(text)
  const spans = findDelimitedMath(normalized)
  const parts: string[] = []
  let cursor = 0
  for (const span of spans) {
    parts.push(texifyPlainSegment(normalized.slice(cursor, span.start)))
    parts.push(emitMathSpan(span))
    cursor = span.end
  }
  parts.push(texifyPlainSegment(normalized.slice(cursor)))
  return parts.join("")
}
