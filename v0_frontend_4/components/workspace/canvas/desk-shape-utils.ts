export function shapeLabel(kind: string) {
  if (kind === "chart") return "Graph"
  if (kind === "figure") return "Figure"
  if (kind === "calculator") return "Calculator"
  return "Card"
}
