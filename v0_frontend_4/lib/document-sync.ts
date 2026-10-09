import { createClient } from "@/lib/supabase/client"

export const DOCUMENTS_BUCKET = "documents"

export function documentObjectPath(userId: string, threadId: string, fileName: string) {
  const cleaned = fileName.replace(/[^\w.\-]+/g, "_").replace(/_+/g, "_").replace(/^[_\.]+|[_\.]+$/g, "")
  const safe = (cleaned || "document.pdf").slice(0, 180)
  return `${userId}/${threadId}/${safe}`
}

export async function uploadSessionPdf(options: {
  userId: string
  threadId: string
  file: File
  accessToken: string
}): Promise<string | null> {
  const supabase = createClient()
  if (!supabase || !options.accessToken) return null
  const objectPath = documentObjectPath(options.userId, options.threadId, options.file.name)
  const { error } = await supabase.storage.from(DOCUMENTS_BUCKET).upload(objectPath, options.file, {
    upsert: true,
    contentType: options.file.type || "application/pdf",
  })
  if (error) {
    console.error("[Documents] Upload failed", error.message)
    return null
  }
  return objectPath
}
