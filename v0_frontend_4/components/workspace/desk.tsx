"use client"

import { BottomDock } from "./bottom-dock"
import { DeskEditor } from "./canvas/desk-editor"
import { DynamicDeskWidget } from "./dynamic-desk-widget"
import { GraphWidget } from "./graph-widget"
import { useDesk } from "./desk-context"

export function Desk() {
  const { cards } = useDesk()
  return (
    <div className="relative flex min-h-0 flex-1 flex-col gap-3 p-4">
      <div className="grid gap-3 md:grid-cols-[minmax(0,1fr)_220px]">
        <DeskEditor />
        <div className="space-y-3">
          {cards.map((card) => (
            <DynamicDeskWidget key={card.id} title={card.title} body={card.body} />
          ))}
          <GraphWidget />
        </div>
      </div>
      <BottomDock />
    </div>
  )
}
