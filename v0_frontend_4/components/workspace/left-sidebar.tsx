"use client"

import { useLanguage } from "./language-context"

export function LeftSidebar({
  collapsed,
  onToggle,
}: {
  collapsed: boolean
  onToggle: () => void
}) {
  const { t } = useLanguage()
  return (
    <aside className={`flex h-full shrink-0 flex-col border-r border-[#e8e4ee] bg-white ${collapsed ? "w-14" : "w-64"}`}>
      <button type="button" onClick={onToggle} className="m-3 rounded-lg px-2 py-1 text-left text-xs font-semibold text-[#797483]">
        {collapsed ? ">" : t("nav.workspaceSections")}
      </button>
      {collapsed ? null : (
        <div className="px-4">
          <p className="text-xs font-bold uppercase tracking-[0.18em] text-[#b52c89]">Chapters</p>
          <ul className="mt-3 space-y-2 text-sm text-[#514c5e]">
            <li className="rounded-xl bg-[#faf0fb] px-3 py-2">Chapter 22 · Rotation</li>
            <li className="rounded-xl px-3 py-2">Chapter 23 · Fields</li>
          </ul>
        </div>
      )}
    </aside>
  )
}
