import type { ChartDeskUpdate } from "@/components/workspace/desk-context"
import type {
  SavedBookmark,
  SavedFormula,
  SavedGraph,
  SavedStruggle,
} from "@/components/workspace/memory-box-context"
import { createClient } from "@/lib/supabase/client"
import { isSyncFrozen, trackPending } from "@/lib/user-data"

export type RemoteMemory = {
  formulas: SavedFormula[]
  graphs: SavedGraph[]
  struggles: SavedStruggle[]
  bookmarks: SavedBookmark[]
}

function iso(savedAt: number) {
  return new Date(savedAt).toISOString()
}

function ms(value: string | null | undefined) {
  const n = value ? Date.parse(value) : NaN
  return Number.isFinite(n) ? n : 0
}

export function mergeById<T extends { id: string; savedAt: number }>(local: T[], remote: T[]): T[] {
  const map = new Map<string, T>()
  for (const item of local) map.set(item.id, item)
  for (const item of remote) {
    const existing = map.get(item.id)
    if (!existing || item.savedAt > existing.savedAt) map.set(item.id, item)
  }
  return [...map.values()].sort((a, b) => b.savedAt - a.savedAt)
}

export async function fetchRemoteMemory(userId: string): Promise<RemoteMemory | null> {
  const supabase = createClient()
  if (!supabase) return null
  const [formulas, graphs, struggles, bookmarks] = await Promise.all([
    supabase.from("memory_formulas").select("id, formula, context, saved_at").eq("user_id", userId),
    supabase.from("memory_graphs").select("id, chart, context, saved_at").eq("user_id", userId),
    supabase
      .from("memory_struggles")
      .select("id, topic, excerpt, question_id, saved_at")
      .eq("user_id", userId),
    supabase.from("memory_bookmarks").select("id, payload, saved_at").eq("user_id", userId),
  ])
  if (formulas.error || graphs.error || struggles.error || bookmarks.error) {
    console.error("[MemoryBox] Remote fetch failed", {
      formulas: formulas.error,
      graphs: graphs.error,
      struggles: struggles.error,
      bookmarks: bookmarks.error,
    })
    return null
  }
  return {
    formulas: (formulas.data ?? []).map((row) => ({
      id: row.id,
      formula: row.formula,
      context: row.context ?? undefined,
      savedAt: ms(row.saved_at),
    })),
    graphs: (graphs.data ?? []).map((row) => ({
      id: row.id,
      chart: row.chart as ChartDeskUpdate,
      context: row.context ?? undefined,
      savedAt: ms(row.saved_at),
    })),
    struggles: (struggles.data ?? []).map((row) => ({
      id: row.id,
      topic: row.topic,
      excerpt: row.excerpt,
      questionId: row.question_id ?? undefined,
      savedAt: ms(row.saved_at),
    })),
    bookmarks: (bookmarks.data ?? []).flatMap((row) => {
      const payload = row.payload as SavedBookmark
      if (!payload || typeof payload !== "object") return []
      return [{ ...payload, id: row.id, savedAt: ms(row.saved_at) }]
    }),
  }
}

function logRemoteError(label: string, error: { message: string } | null) {
  if (error) console.error(`[MemoryBox] ${label}`, error.message)
}

export async function upsertFormula(userId: string, item: SavedFormula) {
  const supabase = createClient()
  if (!supabase || isSyncFrozen()) return
  const { error } = await trackPending(
    Promise.resolve(
      supabase.from("memory_formulas").upsert({
        id: item.id,
        user_id: userId,
        formula: item.formula,
        context: item.context ?? null,
        saved_at: iso(item.savedAt),
      }),
    ),
  )
  logRemoteError("formula upsert", error)
}

export async function upsertGraph(userId: string, item: SavedGraph) {
  const supabase = createClient()
  if (!supabase || isSyncFrozen()) return
  const { error } = await trackPending(
    Promise.resolve(
      supabase.from("memory_graphs").upsert({
        id: item.id,
        user_id: userId,
        chart: item.chart,
        context: item.context ?? null,
        saved_at: iso(item.savedAt),
      }),
    ),
  )
  logRemoteError("graph upsert", error)
}

export async function upsertStruggle(userId: string, item: SavedStruggle) {
  const supabase = createClient()
  if (!supabase || isSyncFrozen()) return
  const { error } = await trackPending(
    Promise.resolve(
      supabase.from("memory_struggles").upsert({
        id: item.id,
        user_id: userId,
        topic: item.topic,
        excerpt: item.excerpt,
        question_id: item.questionId ?? null,
        saved_at: iso(item.savedAt),
      }),
    ),
  )
  logRemoteError("struggle upsert", error)
}

export async function upsertBookmark(userId: string, item: SavedBookmark) {
  const supabase = createClient()
  if (!supabase || isSyncFrozen()) return
  const { error } = await trackPending(
    Promise.resolve(
      supabase.from("memory_bookmarks").upsert({
        id: item.id,
        user_id: userId,
        payload: item,
        saved_at: iso(item.savedAt),
      }),
    ),
  )
  logRemoteError("bookmark upsert", error)
}

export async function deleteRemoteRow(
  table: "memory_formulas" | "memory_graphs" | "memory_struggles" | "memory_bookmarks",
  userId: string,
  id: string,
) {
  const supabase = createClient()
  if (!supabase) return
  const { error } = await supabase.from(table).delete().eq("id", id).eq("user_id", userId)
  logRemoteError(`${table} delete`, error)
}

export async function pushMergedMemory(userId: string, memory: RemoteMemory) {
  await Promise.all([
    ...memory.formulas.map((item) => upsertFormula(userId, item)),
    ...memory.graphs.map((item) => upsertGraph(userId, item)),
    ...memory.struggles.map((item) => upsertStruggle(userId, item)),
    ...memory.bookmarks.map((item) => upsertBookmark(userId, item)),
  ])
}
