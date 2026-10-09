export function DeskCalculatorShape() {
  const keys = ["7", "8", "9", "4", "5", "6", "1", "2", "3", "0", ".", "="]
  return (
    <div className="w-40 rounded-2xl border border-[#e5e1e9] bg-white p-3 shadow-sm">
      <div className="mb-2 h-8 rounded-lg bg-[#f7f6f9]" />
      <div className="grid grid-cols-3 gap-1">
        {keys.map((key) => (
          <button key={key} type="button" className="rounded-md bg-[#f7f6f9] py-1.5 text-xs font-medium text-[#353042]">
            {key}
          </button>
        ))}
      </div>
    </div>
  )
}
