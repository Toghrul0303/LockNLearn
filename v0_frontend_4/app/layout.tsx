import { Analytics } from '@vercel/analytics/next'
import type { Metadata, Viewport } from 'next'
import { Geist, Space_Grotesk } from 'next/font/google'
import './globals.css'
import { LanguageProvider } from '@/components/workspace/language-context'
import { AuthProvider } from '@/components/workspace/auth-context'
import { MemoryBoxProvider } from '@/components/workspace/memory-box-context'
import { MemoryBoxToast } from '@/components/workspace/memory-box-toast'

// Swapped from Inter — Geist Sans reads noticeably heavier/more legible at
// body-copy sizes on the Desk/chat pane while keeping the same clean,
// modern geometric feel.
const geist = Geist({ subsets: ['latin'], variable: '--font-geist' })
const spaceGrotesk = Space_Grotesk({
  subsets: ['latin'],
  variable: '--font-space-grotesk',
})

export const metadata: Metadata = {
  title: 'LockNLearn — AI Academic Workspace',
  description:
    'A focused, AI-powered study workspace for students: task tracking, a smart chat tutor, an interactive graph & scientific keypad, and a distraction-free Focus Mode.',
  icons: {
    icon: [{ url: '/logo.svg', type: 'image/svg+xml' }],
    apple: '/logo.svg',
  },
}

export const viewport: Viewport = {
  colorScheme: 'light dark',
  themeColor: [
    { media: '(prefers-color-scheme: light)', color: '#f4eaf3' },
    { media: '(prefers-color-scheme: dark)', color: '#241521' },
  ],
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html
      lang="en"
      className={`bg-background ${geist.variable} ${spaceGrotesk.variable}`}
    >
      <body className="font-sans antialiased">
        <LanguageProvider>
          <AuthProvider>
            <MemoryBoxProvider>
              {children}
              <MemoryBoxToast />
            </MemoryBoxProvider>
          </AuthProvider>
        </LanguageProvider>
        {process.env.NODE_ENV === 'production' && <Analytics />}
      </body>
    </html>
  )
}
