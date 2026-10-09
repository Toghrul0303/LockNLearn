import { BoardChartPlot } from "./board-chart-plot"
import { BoardFigure } from "./board-figure"
import type { BoardShape } from "./whiteboard-context"
import { cardFrame } from "./canvas-layout"
import { DeskCalculatorShape } from "./desk-calculator-shape"
import { DeskCardShape } from "./desk-card-shape"

export function BoardShapes({ shapes }: { shapes: BoardShape[] }) {
  return (
    <>
      {shapes.map((shape) => {
        const frame = cardFrame(shape.x, shape.y)
        return (
          <div key={shape.id} className="absolute" style={{ left: frame.left, top: frame.top, width: frame.width }}>
            {shape.kind === "chart" ? <BoardChartPlot title={shape.title} /> : null}
            {shape.kind === "figure" ? <BoardFigure title={shape.title} /> : null}
            {shape.kind === "calculator" ? <DeskCalculatorShape /> : null}
            {shape.kind === "card" ? <DeskCardShape title={shape.title} body={shape.body} /> : null}
          </div>
        )
      })}
    </>
  )
}
