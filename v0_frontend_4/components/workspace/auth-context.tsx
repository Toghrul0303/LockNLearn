"use client"

import {
  Fragment,
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react"
import type { Session, User } from "@supabase/supabase-js"
import { createClient } from "@/lib/supabase/client"
import {
  flushPendingWithin,
  readOwner,
  setSyncFrozen,
  wipeLocalUserData,
  writeOwner,
} from "@/lib/user-data"

type AuthContextValue = {
  configured: boolean
  loading: boolean
  user: User | null
  session: Session | null
  signIn: (email: string, password: string) => Promise<string | null>
  signUp: (email: string, password: string) => Promise<string | null>
  signOut: () => Promise<string | null>
}

const AuthContext = createContext<AuthContextValue>({
  configured: false,
  loading: false,
  user: null,
  session: null,
  signIn: async () => null,
  signUp: async () => null,
  signOut: async () => null,
})

/** Longest sign-out waits on the cloud flush, and again on the auth server. */
const SIGN_OUT_STEP_TIMEOUT_MS = 1500
const TIMED_OUT = Symbol("timed-out")

function after(ms: number) {
  return new Promise<typeof TIMED_OUT>((resolve) => {
    window.setTimeout(() => resolve(TIMED_OUT), ms)
  })
}

/**
 * Local data belongs to one account. No owner yet means signed-out work, which the
 * first account to sign in adopts. A different owner means the data is another
 * user's and is removed before anything can sync it.
 */
async function reconcileOwner(userId: string): Promise<boolean> {
  const owner = readOwner()
  if (!owner) {
    writeOwner(userId)
    return false
  }
  if (owner === userId) return false
  await wipeLocalUserData()
  writeOwner(userId)
  return true
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [configured] = useState(() => Boolean(createClient()))
  const [loading, setLoading] = useState(true)
  const [session, setSession] = useState<Session | null>(null)
  const [authEpoch, setAuthEpoch] = useState(0)

  useEffect(() => {
    const supabase = createClient()
    if (!supabase) {
      setLoading(false)
      return
    }
    let cancelled = false
    let applySeq = 0
    const apply = async (next: Session | null) => {
      const seq = ++applySeq
      let wiped = false
      if (next?.user) wiped = await reconcileOwner(next.user.id)
      if (cancelled || seq !== applySeq) return
      if (wiped) setAuthEpoch((n) => n + 1)
      setSession(next)
      setLoading(false)
    }
    void supabase.auth.getSession().then(({ data }) => apply(data.session))
    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, next) => {
      void apply(next)
    })
    return () => {
      cancelled = true
      subscription.unsubscribe()
    }
  }, [])

  const signIn = useCallback(async (email: string, password: string) => {
    const supabase = createClient()
    if (!supabase) return "Supabase is not configured."
    const { error } = await supabase.auth.signInWithPassword({ email, password })
    return error?.message ?? null
  }, [])

  const signUp = useCallback(async (email: string, password: string) => {
    const supabase = createClient()
    if (!supabase) return "Supabase is not configured."
    const { error } = await supabase.auth.signUp({ email, password })
    return error?.message ?? null
  }, [])

  const signOut = useCallback(async () => {
    const supabase = createClient()
    if (!supabase) return "Supabase is not configured."
    await flushPendingWithin(SIGN_OUT_STEP_TIMEOUT_MS)
    setSyncFrozen(true)
    let message: string | null = null
    try {
      const outcome = await Promise.race([
        supabase.auth.signOut().then(({ error }) => error?.message ?? null),
        after(SIGN_OUT_STEP_TIMEOUT_MS),
      ])
      if (outcome === TIMED_OUT) setSession(null)
      else message = outcome
    } catch (error) {
      message = error instanceof Error ? error.message : "Sign out failed."
    }
    try {
      await wipeLocalUserData()
    } finally {
      setAuthEpoch((n) => n + 1)
    }
    return message
  }, [])

  // Writers stay frozen until the old tree has unmounted, so its cleanup flushes cannot
  // write the previous user's data back after the wipe.
  useEffect(() => {
    setSyncFrozen(false)
  }, [authEpoch])

  const value = useMemo<AuthContextValue>(
    () => ({
      configured,
      loading,
      user: session?.user ?? null,
      session,
      signIn,
      signUp,
      signOut,
    }),
    [configured, loading, session, signIn, signUp, signOut],
  )

  return (
    <AuthContext.Provider value={value}>
      <Fragment key={authEpoch}>{children}</Fragment>
    </AuthContext.Provider>
  )
}

export function useAuth() {
  return useContext(AuthContext)
}
