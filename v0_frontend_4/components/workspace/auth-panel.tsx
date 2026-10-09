"use client"

import { useState, type FormEvent } from "react"
import { useRouter } from "next/navigation"
import { cn } from "@/lib/utils"
import { useAuth } from "./auth-context"
import { useLanguage } from "./language-context"

export function LoginForm() {
  const router = useRouter()
  const { configured, signIn, signUp } = useAuth()
  const { t } = useLanguage()
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [info, setInfo] = useState<string | null>(null)

  const run = async (mode: "in" | "up", email: string, password: string) => {
    setBusy(true)
    setError(null)
    setInfo(null)
    if (!email || !password) {
      setBusy(false)
      setError(t("sidebar.enterCredentials"))
      return
    }
    const message = mode === "in" ? await signIn(email, password) : await signUp(email, password)
    setBusy(false)
    if (message) {
      setError(message)
      return
    }
    if (mode === "up") setInfo(t("sidebar.confirmEmail"))
    else router.replace("/workspace")
  }

  const onSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    const submitter = (event.nativeEvent as SubmitEvent).submitter
    const data = new FormData(event.currentTarget, submitter)
    const email = String(data.get("email") ?? "").trim()
    const password = String(data.get("password") ?? "")
    const mode = data.get("mode") === "up" ? "up" : "in"
    void run(mode, email, password)
  }

  if (!configured) {
    return <p className="text-sm text-muted-foreground">{t("sidebar.authUnavailable")}</p>
  }

  return (
    <form onSubmit={onSubmit} className="flex w-full max-w-sm flex-col gap-3">
      <input
        type="email"
        name="email"
        autoComplete="email"
        required
        placeholder={t("sidebar.email")}
        className="h-10 rounded-xl border border-border bg-background px-3 text-sm outline-none placeholder:text-muted-foreground focus:border-primary/50"
      />
      <input
        type="password"
        name="password"
        autoComplete="current-password"
        required
        minLength={6}
        placeholder={t("sidebar.password")}
        className="h-10 rounded-xl border border-border bg-background px-3 text-sm outline-none placeholder:text-muted-foreground focus:border-primary/50"
      />
      {error ? <p className="text-xs leading-snug text-destructive">{error}</p> : null}
      {info ? <p className="text-xs leading-snug text-muted-foreground">{info}</p> : null}
      <div className="flex gap-2">
        <button
          type="submit"
          name="mode"
          value="in"
          disabled={busy}
          className={cn(
            "bg-brand-gradient h-10 flex-1 rounded-xl text-sm font-semibold text-white",
            "disabled:cursor-not-allowed disabled:opacity-50",
          )}
        >
          {t("sidebar.signIn")}
        </button>
        <button
          type="submit"
          name="mode"
          value="up"
          disabled={busy}
          className="h-10 flex-1 rounded-xl border border-border bg-card text-sm font-semibold text-muted-foreground hover:text-foreground disabled:cursor-not-allowed disabled:opacity-50"
        >
          {t("sidebar.signUp")}
        </button>
      </div>
    </form>
  )
}
