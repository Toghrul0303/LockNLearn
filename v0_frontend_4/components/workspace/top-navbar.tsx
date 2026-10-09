"use client"

import Link from "next/link"
import { Globe, Moon, Sun } from "lucide-react"
import { cn } from "@/lib/utils"
import { NAV_TABS } from "./data"
import { useMode } from "./mode-context"
import { useLanguage } from "./language-context"
import { LOCALES, type Locale } from "@/lib/i18n"

function Logo() {
  return (
    // eslint-disable-next-line @next/next/no-img-element
    <img
      src="/logo-right.svg"
      alt="LockNLearn"
      className="h-9 max-w-none shrink-0 overflow-visible object-contain"
      style={{
        width: "auto",
        objectViewBox: "inset(38.76% 7.66% 40.27% 9.89%)",
      }}
    />
  )
}

function Tabs({ vertical }: { vertical?: boolean }) {
  const { t } = useLanguage()
  return (
    <nav
      aria-label={t("nav.workspaceSections")}
      className={cn(
        "flex items-center gap-1",
        vertical && "flex-col items-stretch gap-0.5",
      )}
    >
      <Link
        href="/"
        className={cn(
          "rounded-lg px-3 py-1.5 text-sm font-medium text-muted-foreground transition-colors hover:bg-accent/60 hover:text-foreground",
          vertical && "text-left",
        )}
      >
        {t("nav.home")}
      </Link>
      {NAV_TABS.map((tab) => (
        <Link
          key={tab}
          href="/workspace"
          className={cn(
            "rounded-lg bg-accent px-3 py-1.5 text-sm font-medium text-accent-foreground transition-colors",
            vertical && "text-left",
          )}
        >
          {t(`nav.${tab}`)}
        </Link>
      ))}
    </nav>
  )
}

function ModeToggle() {
  const { dark, toggleTheme } = useMode()
  const { t } = useLanguage()
  return (
    <button
      type="button"
      onClick={toggleTheme}
      aria-pressed={dark}
      aria-label={dark ? t("nav.dark") : t("nav.light")}
      className="group inline-flex items-center gap-2 rounded-full border border-border bg-card px-1 py-1 pr-3 text-sm font-medium shadow-sm transition-colors hover:border-primary/40"
    >
      <span
        className={cn(
          "grid size-6 place-items-center rounded-full",
          dark ? "bg-brand-gradient text-white" : "bg-secondary text-muted-foreground",
        )}
      >
        {dark ? (
          <Moon className="size-3.5" aria-hidden="true" />
        ) : (
          <Sun className="size-3.5" aria-hidden="true" />
        )}
      </span>
      <span className="hidden md:inline">{dark ? t("nav.dark") : t("nav.light")}</span>
    </button>
  )
}

function LanguageSwitcher() {
  const { locale, setLocale, t } = useLanguage()
  return (
    <label className="relative inline-flex items-center">
      <span className="sr-only">{t("nav.language")}</span>
      <Globe className="pointer-events-none absolute left-2.5 size-3.5 text-muted-foreground" aria-hidden="true" />
      <select
        value={locale}
        aria-label={t("nav.language")}
        onChange={(event) => setLocale(event.target.value as Locale)}
        className="h-8 cursor-pointer appearance-none rounded-full border border-border bg-card py-0 pr-7 pl-8 text-xs font-semibold uppercase tracking-wide text-foreground shadow-sm outline-none transition-colors hover:border-primary/40"
      >
        {LOCALES.map((item) => (
          <option key={item.code} value={item.code}>
            {item.short}
          </option>
        ))}
      </select>
    </label>
  )
}

export function TopNavbar() {
  return (
    <header className="relative z-30 flex h-16 shrink-0 items-center border-b border-border px-4">
      <div className="flex items-center gap-2">
        <Tabs />
      </div>

      <div className="absolute top-1/2 left-1/2 flex -translate-x-1/2 -translate-y-1/2 items-center">
        <Logo />
      </div>

      <div className="ml-auto flex items-center gap-2">
        <LanguageSwitcher />
        <ModeToggle />
      </div>
    </header>
  )
}
