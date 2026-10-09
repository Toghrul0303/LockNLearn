import { cn } from "@/lib/utils"

export function TutorAvatar({ className }: { className?: string }) {
  return (
    // eslint-disable-next-line @next/next/no-img-element
    <img
      src="/AI%20Avatar.png"
      alt=""
      className={cn("shrink-0 rounded-lg object-contain", className)}
    />
  )
}
