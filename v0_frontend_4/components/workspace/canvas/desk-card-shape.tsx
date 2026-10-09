import { BoardMath } from "./board-math"
import { shapeLabel } from "./desk-shape-utils"

export function DeskCardShape({ title, body }: { title: string; body: string }) {
  return (
    <article className="rounded-2xl border border-[#e5e1e9] bg-white p-4 shadow-sm">
      <p className="text-[11px] font-semibold uppercase tracking-wide text-[#b52c89]">{shapeLabel("card")}</p>
      <h3 className="mt-1 text-sm font-semibold text-[#353042]">{title}</h3>
      <div className="mt-2">
        <BoardMath text={body} />
      </div>
    </article>
  )
}
