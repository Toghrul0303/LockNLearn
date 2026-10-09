export function DynamicDeskWidget({ title, body }: { title: string; body: string }) {
  return (
    <article className="rounded-2xl border border-[#e5e1e9] bg-white p-4 shadow-sm">
      <h3 className="text-sm font-semibold text-[#353042]">{title}</h3>
      <p className="mt-2 text-sm leading-6 text-[#777181]">{body}</p>
    </article>
  )
}
