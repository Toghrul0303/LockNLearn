export function BoardFigure({ title }: { title: string }) {
  return (
    <div className="grid h-24 place-items-center rounded-xl bg-[#f7fbff] text-xs text-[#797483]">
      {title}
    </div>
  )
}
