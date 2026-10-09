"use client"

import { useState } from "react"
import { Calculator, Delete, X } from "lucide-react"
import { cn } from "@/lib/utils"
import { useLanguage } from "./language-context"

const KEYS = [
  ["C", "⌫", "%", "÷"],
  ["7", "8", "9", "×"],
  ["4", "5", "6", "−"],
  ["1", "2", "3", "+"],
  ["±", "0", ".", "="],
] as const

function toEvalExpr(expr: string) {
  return expr.replace(/×/g, "*").replace(/÷/g, "/").replace(/−/g, "-")
}

function evaluate(expr: string): string {
  const sanitized = toEvalExpr(expr).replace(/[^0-9.+\-*/%() ]/g, "")
  if (!sanitized.trim()) return "0"
  try {
    const result = evalArithmetic(sanitized)
    if (!Number.isFinite(result)) return "Error"
    return String(Number(result.toPrecision(12)))
  } catch {
    return "Error"
  }
}

/** Basic + − × ÷ % parser — no eval / Function. */
function evalArithmetic(input: string): number {
  const tokens = input.match(/\d+\.\d+|\d+\.|\.\d+|\d+|[+\-*/%()]/g)
  if (!tokens) throw new Error("empty")
  let i = 0
  const peek = () => tokens[i]
  const eat = (expected?: string) => {
    const token = tokens[i]
    if (expected && token !== expected) throw new Error("unexpected")
    i += 1
    return token
  }
  const parseExpr = (): number => {
    let value = parseTerm()
    while (peek() === "+" || peek() === "-") {
      const op = eat()
      const rhs = parseTerm()
      value = op === "+" ? value + rhs : value - rhs
    }
    return value
  }
  const parseTerm = (): number => {
    let value = parseFactor()
    while (peek() === "*" || peek() === "/" || peek() === "%") {
      const op = eat()
      const rhs = parseFactor()
      if (op === "*") value *= rhs
      else if (op === "/") value /= rhs
      else value %= rhs
    }
    return value
  }
  const parseFactor = (): number => {
    if (peek() === "+") {
      eat()
      return parseFactor()
    }
    if (peek() === "-") {
      eat()
      return -parseFactor()
    }
    if (peek() === "(") {
      eat("(")
      const value = parseExpr()
      eat(")")
      return value
    }
    const token = eat()
    if (!token || Number.isNaN(Number(token))) throw new Error("number")
    return Number(token)
  }
  const value = parseExpr()
  if (i !== tokens.length) throw new Error("trailing")
  return value
}

export function CalculatorKeypad({ onClose }: { onClose?: () => void }) {
  const { t } = useLanguage()
  const [expr, setExpr] = useState("0")
  const [fresh, setFresh] = useState(true)

  const press = (key: string) => {
    if (key === "C") {
      setExpr("0")
      setFresh(true)
      return
    }
    if (key === "⌫") {
      setExpr((prev) => (prev.length <= 1 || prev === "Error" ? "0" : prev.slice(0, -1)))
      setFresh(false)
      return
    }
    if (key === "=") {
      setExpr(evaluate(expr))
      setFresh(true)
      return
    }
    if (key === "±") {
      setExpr((prev) => {
        if (prev.startsWith("-")) return prev.slice(1) || "0"
        if (prev === "0" || prev === "Error") return prev
        return `-${prev}`
      })
      return
    }
    const isOp = ["÷", "×", "−", "+", "%"].includes(key)
    setExpr((prev) => {
      if (prev === "Error") return isOp ? "0" : key
      if (fresh && !isOp) return key === "." ? "0." : key
      if (prev === "0" && key !== ".") return key
      return prev + key
    })
    setFresh(false)
  }

  return (
    <div
      className="w-[220px] rounded-2xl border border-border bg-popover/95 p-3 shadow-2xl backdrop-blur-md"
      onPointerDown={(event) => event.stopPropagation()}
    >
      <div className="mb-2 flex items-center gap-2">
        <span className="bg-brand-gradient grid size-7 place-items-center rounded-lg text-white">
          <Calculator className="size-3.5" aria-hidden="true" />
        </span>
        <h3 className="font-display text-sm font-semibold">{t("desk.calculator")}</h3>
        {onClose ? (
          <button
            type="button"
            onClick={onClose}
            aria-label={t("desk.closePanel")}
            className="ml-auto grid size-6 place-items-center rounded-md text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
          >
            <X className="size-3.5" aria-hidden="true" />
          </button>
        ) : null}
      </div>
      <div className="mb-2 flex min-h-10 items-center justify-end rounded-xl bg-secondary/70 px-3 py-2">
        <span className="truncate font-mono text-lg font-medium tabular-nums text-foreground">
          {expr}
        </span>
      </div>
      <div className="grid grid-cols-4 gap-1.5">
        {KEYS.flat().map((key) => {
          const op = ["÷", "×", "−", "+", "=", "%"].includes(key)
          const danger = key === "C"
          return (
            <button
              key={key}
              type="button"
              aria-label={key === "⌫" ? "Backspace" : key}
              onClick={() => press(key)}
              className={cn(
                "grid h-10 place-items-center rounded-lg text-sm font-medium transition-colors",
                op && "bg-primary/12 text-primary hover:bg-primary/20",
                danger && "text-destructive hover:bg-destructive/10",
                !op && !danger && "bg-secondary/60 text-foreground hover:bg-accent",
              )}
            >
              {key === "⌫" ? <Delete className="size-4" aria-hidden="true" /> : key}
            </button>
          )
        })}
      </div>
    </div>
  )
}
