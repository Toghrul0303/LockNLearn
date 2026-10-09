import { DeskCardShapeUtil } from "./desk-card-shape"
import { BoardFigureShapeUtil } from "./board-figure"
import { DeskCalculatorShapeUtil } from "./desk-calculator-shape"
import {
  BoardBranchShapeUtil,
  BoardChartShapeUtil,
  BoardQuestionShapeUtil,
  BoardResultShapeUtil,
  BoardStepShapeUtil,
} from "./board-shapes"

export const deskShapeUtils = [
  DeskCardShapeUtil,
  BoardQuestionShapeUtil,
  BoardStepShapeUtil,
  BoardResultShapeUtil,
  BoardChartShapeUtil,
  BoardFigureShapeUtil,
  BoardBranchShapeUtil,
  DeskCalculatorShapeUtil,
]
