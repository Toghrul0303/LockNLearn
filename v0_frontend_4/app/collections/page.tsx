import { Suspense } from "react"
import { CollectionsPage } from "@/components/collections/collections-page"

export default function Page() {
  return (
    <Suspense fallback={null}>
      <CollectionsPage />
    </Suspense>
  )
}
