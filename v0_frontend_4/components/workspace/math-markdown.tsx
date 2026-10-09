"use client"

import {
  memo,
  useLayoutEffect,
  useRef,
  type ElementType,
  type ReactNode,
} from "react"
import ReactMarkdown, { type Options as MarkdownOptions } from "react-markdown"

type MathJaxApi = {
  typesetPromise?: (elements?: Element[]) => Promise<void>
  typesetClear?: (elements?: Element[]) => void
  startup?: { promise?: Promise<void> }
}

type WindowWithMathJax = Window & {
  MathJax?: MathJaxApi
  __locknlearnMathJax?: Promise<void>
}

function ensureMathJaxSvg(): Promise<void> {
  if (typeof window === "undefined") return Promise.resolve()
  const w = window as WindowWithMathJax
  if (w.MathJax?.typesetPromise) {
    return w.MathJax.startup?.promise ?? Promise.resolve()
  }
  if (w.__locknlearnMathJax) return w.__locknlearnMathJax

  // Config must be on `window.MathJax` BEFORE the es5 bundle loads.
  ;(window as unknown as { MathJax: Record<string, unknown> }).MathJax = {
    tex: {
      inlineMath: [["\\(", "\\)"]],
      displayMath: [["\\[", "\\]"]],
    },
    svg: { fontCache: "local" },
    options: {
      menuOptions: {
        settings: { assistiveMml: true },
      },
    },
    startup: { typeset: false },
  }

  w.__locknlearnMathJax = import("mathjax-full/es5/tex-svg.js").then(() => {
    return (window as WindowWithMathJax).MathJax?.startup?.promise ?? Promise.resolve()
  })
  return w.__locknlearnMathJax
}

function sourceHasTeX(source: string): boolean {
  return /\$|\\\[|\\\(|\\begin\{/.test(source)
}

type MathMarkdownProps = MarkdownOptions & {
  className?: string
  as?: "div" | "span"
  children: ReactNode
}

/**
 * Client markdown renderer that typesets TeX to MathJax SVG after mount.
 *
 * MathJax rewrites real DOM nodes into `<mjx-container>` SVGs. If React later
 * reconciles that subtree (streaming chunks, study-mode rerenders, a new
 * Highlight-to-Ask bubble), it calls removeChild on nodes MathJax already
 * replaced → `NotFoundError: node to be removed is not a child`.
 *
 * Guardrails:
 * 1. Remount the markdown host when the source string changes (`key`) so
 *    React tears down a whole node instead of diffing into MathJax's tree.
 * 2. Skip re-render when the source is unchanged (`memo`) so sibling updates
 *    don't touch already-typeset bubbles.
 * 3. `typesetClear` the captured host on effect cleanup; ignore stale
 *    `typesetPromise` results after unmount.
 */
function MathMarkdownInner({
  className,
  as = "div",
  children,
  ...markdownProps
}: MathMarkdownProps) {
  const hostRef = useRef<HTMLElement | null>(null)
  const source = typeof children === "string" ? children : String(children ?? "")
  const InnerTag = as === "span" ? "span" : "div"

  useLayoutEffect(() => {
    const el = hostRef.current
    if (!el || !sourceHasTeX(source)) return
    let cancelled = false
    let timer: ReturnType<typeof setTimeout> | undefined

    const run = async () => {
      await ensureMathJaxSvg()
      if (cancelled) return
      const mathjax = (window as WindowWithMathJax).MathJax
      try {
        await mathjax?.typesetPromise?.([el])
      } catch {
        // Incomplete TeX while a chat turn is still streaming is expected.
      }
      if (cancelled) {
        try {
          mathjax?.typesetClear?.([el])
        } catch {
          // Host may already be detached.
        }
      }
    }

    // Streaming appends many times per second; wait for a pause so we typeset
    // the latest source instead of racing half-closed `$` / `\(` delimiters.
    timer = setTimeout(() => {
      void run()
    }, 80)

    return () => {
      cancelled = true
      if (timer) clearTimeout(timer)
      const mathjax = (window as WindowWithMathJax).MathJax
      try {
        mathjax?.typesetClear?.([el])
      } catch {
        // Host may already be detached.
      }
    }
  }, [source])

  const Tag = as as ElementType
  const setHost = (node: HTMLElement | null) => {
    hostRef.current = node
  }
  return (
    <Tag className={className}>
      <InnerTag ref={setHost} key={source}>
        <ReactMarkdown {...markdownProps}>{source}</ReactMarkdown>
      </InnerTag>
    </Tag>
  )
}

export const MathMarkdown = memo(MathMarkdownInner, (prev, next) => {
  return (
    prev.children === next.children &&
    prev.className === next.className &&
    prev.as === next.as
  )
})
MathMarkdown.displayName = "MathMarkdown"
