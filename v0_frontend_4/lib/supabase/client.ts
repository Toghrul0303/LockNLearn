import { createBrowserClient } from "@supabase/ssr"
import { getSupabasePublicEnv } from "./env"

export function createClient() {
  const env = getSupabasePublicEnv()
  if (!env) return null
  return createBrowserClient(env.url, env.anonKey)
}
