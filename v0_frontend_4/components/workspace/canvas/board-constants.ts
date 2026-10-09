export const CARD_WIDTH = 280
export const CARD_MIN_HEIGHT = 120

export function placeCard(index: number) {
  return { x: 40 + (index % 3) * (CARD_WIDTH + 24), y: 36 + Math.floor(index / 3) * 160 }
}
