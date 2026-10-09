"use client"

import Link from "next/link"
import { LoginForm } from "@/components/workspace/auth-panel"
import { useAuth } from "@/components/workspace/auth-context"
import { useLanguage } from "@/components/workspace/language-context"

export default function LoginPage() {
  const { user, loading, signOut } = useAuth()
  const { t } = useLanguage()

  return (
    <main className="grid min-h-dvh place-items-center bg-background px-4 text-foreground">
      <div className="flex w-full max-w-md flex-col items-center gap-6 rounded-2xl border border-border bg-card p-8 shadow-sm">
        <div className="text-center">
          <h1 className="font-display text-xl font-semibold">{t("sidebar.signIn")}</h1>
          {!loading && user ? (
            <p className="mt-1 text-sm text-muted-foreground">
              {t("sidebar.signedInAs", { email: user.email ?? "" })}
            </p>
          ) : (
            <p className="mt-1 text-sm text-muted-foreground">{t("sidebar.authSubtitle")}</p>
          )}
        </div>
        {loading ? null : user ? (
          <div className="flex w-full max-w-sm flex-col gap-2">
            <Link
              href="/workspace"
              className="bg-brand-gradient grid h-10 place-items-center rounded-xl text-sm font-semibold text-white"
            >
              {t("sidebar.goToWorkspace")}
            </Link>
            <button
              type="button"
              onClick={() => void signOut()}
              className="h-10 rounded-xl border border-border bg-card text-sm font-semibold text-muted-foreground hover:text-foreground"
            >
              {t("sidebar.signOut")}
            </button>
          </div>
        ) : (
          <LoginForm />
        )}
      </div>
    </main>
  )
}
