import type { Metadata, Viewport } from 'next'
import { Geist, Space_Grotesk } from 'next/font/google'
import './globals.css'
import { LanguageProvider } from '@/components/workspace/language-context'
import { AuthProvider } from '@/components/workspace/auth-context'

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
          <AuthProvider>{children}</AuthProvider>
        </LanguageProvider>
      </body>
    </html>
  )
}
