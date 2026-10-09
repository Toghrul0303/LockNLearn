import { CARD_WIDTH } from "./board-constants"

export function cardFrame(x: number, y: number) {
  return { left: x, top: y, width: CARD_WIDTH }
}
