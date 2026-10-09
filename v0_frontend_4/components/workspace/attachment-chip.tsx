export function AttachmentChip({ name }: { name: string }) {
  return (
    <span className="inline-flex items-center rounded-full border border-[#eadfeb] bg-white px-2.5 py-1 text-[11px] font-medium text-[#514c5e]">
      {name}
    </span>
  )
}
