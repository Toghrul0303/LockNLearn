'use client'

import { useState } from 'react'
import {
  ArrowRight,
  BookOpen,
  Calculator,
  Check,
  ChevronDown,
  Clock3,
  Globe2,
  GraduationCap,
  Lightbulb,
  ListChecks,
  Menu,
  MousePointer2,
  NotebookTabs,
  PanelTop,
  Sparkles,
  Target,
  X,
} from 'lucide-react'

const features = [
  {
    title: 'Smart Board & AI Tutor',
    description: 'Work in parallel with your AI tutor while solving problems. Get clear, step-by-step explanations.',
    icon: Sparkles,
    className: 'md:col-span-2 md:row-span-2',
    tone: 'bg-[#fff8fc]',
  },
  {
    title: 'Explain Tool',
    description: 'Select any confusing part and get an instant, targeted explanation.',
    icon: Lightbulb,
    className: 'md:col-span-1',
    tone: 'bg-[#f7fbff]',
  },
  {
    title: 'Memory Box',
    description: 'Save golden rules and solutions with one tap, and retrieve them instantly when needed.',
    icon: NotebookTabs,
    className: 'md:col-span-1',
    tone: 'bg-[#fffaf5]',
  },
  {
    title: 'Smart Task Tracker',
    description: 'Set up your problem list, track your progress, and strengthen your weak points.',
    icon: ListChecks,
    className: 'md:col-span-2',
    tone: 'bg-[#f8f8ff]',
  },
  {
    title: 'Focus Toolkit',
    description: 'No distractions. Expand the whiteboard to full screen, and manage your time with the built-in calculator and Pomodoro timer.',
    icon: Target,
    className: 'md:col-span-2',
    tone: 'bg-[#f7fcfa]',
  },
]

const plans = [
  { name: 'Free', price: '$0', detail: 'For getting started', features: ['Daily query limit', '1 workspace', 'Core whiteboard tools'] },
  { name: 'Pro', price: '$5', detail: 'For focused learners', features: ['Unlimited queries', 'Persistent workspace', 'AI-powered explanations', 'Saved solutions & memory'], featured: true },
  { name: 'Premium', price: '$15', detail: 'For serious STEM study', features: ['Everything in Pro', 'AI vision analysis', 'Advanced study insights', 'Priority support'] },
]

export default function Page() {
  const [menuOpen, setMenuOpen] = useState(false)
  const [languageOpen, setLanguageOpen] = useState(false)
  const [language, setLanguage] = useState('EN')

  return (
    <main className="min-h-screen overflow-hidden bg-[#fbfbfd] text-[#292638]">
      <header className="sticky top-0 z-50 border-b border-[#ebe8f0]/90 bg-[#fbfbfd]/90 backdrop-blur-xl">
        <div className="mx-auto flex h-[72px] max-w-7xl items-center justify-between px-5 sm:px-8 lg:px-10">
          <a href="#top" className="flex items-center" aria-label="LockNLearn home">
            <img src="/logo-right.svg" alt="LockNLearn" className="h-9 max-w-none w-auto shrink-0 overflow-visible object-contain" style={{ objectViewBox: "inset(38.76% 7.66% 40.27% 9.89%)" }} />
          </a>
          <button className="rounded-lg p-2 text-[#6d687b] md:hidden" onClick={() => setMenuOpen(!menuOpen)} aria-label="Toggle menu">
            {menuOpen ? <X size={21} /> : <Menu size={21} />}
          </button>
          <nav className={`${menuOpen ? 'flex' : 'hidden'} absolute left-4 right-4 top-[82px] flex-col gap-5 rounded-2xl border border-[#e8e4ee] bg-white p-5 shadow-xl md:static md:flex md:flex-row md:items-center md:gap-8 md:border-0 md:bg-transparent md:p-0 md:shadow-none`}>
            <a href="#top" className="text-sm font-medium text-[#a82e83]">Homepage</a>
            <a href="/workspace" className="text-sm font-medium text-[#706b7c] transition-colors hover:text-[#a82e83]">Workspace</a>
            <div className="relative">
              <button onClick={() => setLanguageOpen(!languageOpen)} className="flex items-center gap-2 text-sm font-medium text-[#706b7c]" aria-expanded={languageOpen}>
                <Globe2 size={16} /> {language} <ChevronDown size={14} />
              </button>
              {languageOpen && <div className="absolute right-0 top-8 z-10 w-24 rounded-xl border border-[#e8e4ee] bg-white p-1.5 shadow-lg">{['EN', 'AZ', 'TR', 'RU'].map((item) => <button key={item} onClick={() => { setLanguage(item); setLanguageOpen(false) }} className="block w-full rounded-lg px-3 py-2 text-left text-xs font-semibold hover:bg-[#faf0fb]">{item}</button>)}</div>}
            </div>
            <a href="/login" className="inline-flex items-center justify-center rounded-full bg-[#bf2b91] px-5 py-2.5 text-sm font-semibold text-white shadow-[0_7px_18px_rgba(183,40,135,0.22)] transition hover:-translate-y-0.5 hover:bg-[#a8227e]">Login</a>
          </nav>
        </div>
      </header>

      <section id="top" className="relative px-5 pb-12 pt-20 sm:px-8 sm:pt-28 lg:px-10 lg:pt-32">
        <div className="pointer-events-none absolute left-1/2 top-0 -z-0 h-[440px] w-[760px] -translate-x-1/2 rounded-full bg-[#f7e8f7] opacity-70 blur-3xl" />
        <div className="relative z-10 mx-auto max-w-4xl text-center">
          <div className="mx-auto mb-6 flex w-fit items-center gap-2 rounded-full border border-[#eadfeb] bg-white px-4 py-2 text-xs font-semibold text-[#a52a82] shadow-sm"><Sparkles size={14} /> Built for curious minds</div>
          <h1 className="mx-auto max-w-3xl text-balance text-4xl font-semibold leading-[1.08] tracking-[-0.055em] text-[#262334] sm:text-6xl lg:text-7xl">YOUR INTERACTIVE <span className="text-[#b52d8b]">STEM</span> WORKSPACE</h1>
          <p className="mx-auto mt-7 max-w-2xl text-base leading-7 text-[#716d7d] sm:text-lg">You don&apos;t need any other app. Grasp complex problems with a smart whiteboard and step-by-step AI guidance. Say goodbye to opening 10 different tabs.</p>
          <div className="mt-9 flex flex-col justify-center gap-3 sm:flex-row"><a href="/workspace" className="group inline-flex items-center justify-center gap-2 rounded-full bg-[#ba2b8a] px-6 py-3.5 text-sm font-semibold text-white shadow-[0_10px_24px_rgba(184,41,137,0.24)] transition hover:-translate-y-1 hover:bg-[#a7247e]">Start Learning <ArrowRight size={16} className="transition group-hover:translate-x-0.5" /></a><a href="#features" className="inline-flex items-center justify-center rounded-full border border-[#dcd7e3] bg-white px-6 py-3.5 text-sm font-semibold text-[#514c5e] shadow-sm transition hover:-translate-y-1 hover:border-[#ba2b8a] hover:text-[#a7247e]">How it works</a></div>
        </div>
      </section>

      <section id="workspace" className="px-5 pb-24 sm:px-8 lg:px-10 lg:pb-32">
        <div className="mx-auto max-w-6xl rounded-[22px] border border-[#e2dfe8] bg-white p-2 shadow-[0_28px_70px_rgba(48,38,61,0.13)] sm:p-3">
          <div className="flex h-9 items-center gap-1.5 rounded-t-[14px] border-b border-[#efedf2] bg-[#faf9fc] px-4"><span className="size-2.5 rounded-full bg-[#f2a0a5]" /><span className="size-2.5 rounded-full bg-[#f2ce8d]" /><span className="size-2.5 rounded-full bg-[#9ed8ae]" /><div className="mx-auto h-5 w-2/5 rounded-md bg-white shadow-inner sm:w-1/3" /></div>
          <div className="relative aspect-video overflow-hidden rounded-b-[14px] bg-[#f8f8fa]"><img src="/Desk.png" alt="LockNLearn interactive STEM workspace" className="absolute inset-0 size-full object-cover" /></div>
        </div>
      </section>

      <section id="features" className="bg-[#f7f6f9] px-5 py-20 sm:px-8 lg:px-10 lg:py-28"><div className="mx-auto max-w-6xl"><div className="mb-12 flex flex-col justify-between gap-5 sm:flex-row sm:items-end"><div><p className="mb-3 text-xs font-bold uppercase tracking-[0.22em] text-[#b52c89]">Everything in one place</p><h2 className="text-3xl font-semibold tracking-[-0.045em] sm:text-5xl">Study smarter, together.</h2></div><p className="max-w-sm text-sm leading-6 text-[#797483]">One calm, focused space for understanding difficult ideas and building lasting confidence.</p></div><div className="grid auto-rows-[minmax(190px,auto)] gap-4 md:grid-cols-4">{features.map(({ title, description, icon: Icon, className, tone }) => <article key={title} className={`${className} ${tone} group relative overflow-hidden rounded-3xl border border-[#e5e1e9] p-6 shadow-sm transition-all duration-300 hover:-translate-y-1 hover:shadow-lg sm:p-8`}>
            {title === 'Smart Board & AI Tutor' && <div className="pointer-events-none absolute inset-y-0 right-0 w-1/2 opacity-75 transition-opacity duration-500 group-hover:opacity-95" style={{ maskImage: 'linear-gradient(to left, black 0%, black 48%, transparent 100%)', WebkitMaskImage: 'linear-gradient(to left, black 0%, black 48%, transparent 100%)' }}><img src="/Chatbot.png" alt="" className="size-full object-cover object-left mix-blend-multiply" /></div>}
            {title === 'Focus Toolkit' && <div className="pointer-events-none absolute bottom-0 right-0 h-[78%] w-[62%] translate-x-12 translate-y-6 opacity-60" style={{ maskImage: 'linear-gradient(to top left, black 0%, black 40%, transparent 100%)', WebkitMaskImage: 'linear-gradient(to top left, black 0%, black 40%, transparent 100%)' }}><img src="/Focus%20Toolkit.png" alt="" className="size-full object-cover object-left-top mix-blend-multiply" /></div>}
            <div className={`relative z-10 ${title === 'Smart Board & AI Tutor' || title === 'Focus Toolkit' ? 'max-w-[48%]' : ''}`}><div className={`${title === 'Smart Board & AI Tutor' || title === 'Focus Toolkit' ? 'mb-12' : 'mb-6'} flex size-11 items-center justify-center rounded-2xl bg-white text-[#b52c89] shadow-sm`}><Icon size={21} /></div><h3 className="text-xl font-semibold tracking-[-0.03em] text-[#353042]">{title}</h3><p className="mt-3 max-w-md text-sm leading-6 text-[#777181]">{description}</p></div><div className="absolute -bottom-12 -right-8 size-36 rounded-full border-[16px] border-white/50 transition-transform duration-500 group-hover:scale-125" /></article>)}</div></div></section>

      <section id="pricing" className="px-5 py-20 sm:px-8 lg:px-10 lg:py-32"><div className="mx-auto max-w-6xl"><div className="mx-auto mb-12 max-w-xl text-center"><p className="mb-3 text-xs font-bold uppercase tracking-[0.22em] text-[#b52c89]">Simple plans</p><h2 className="text-3xl font-semibold tracking-[-0.045em] sm:text-5xl">Choose your pace.</h2><p className="mt-4 text-sm leading-6 text-[#797483]">Start for free, then unlock more room to think when you&apos;re ready.</p></div><div className="grid gap-4 lg:grid-cols-3">{plans.map((plan) => <article key={plan.name} className={`relative rounded-3xl border p-7 ${plan.featured ? 'border-[#c33a98] bg-[#fffafd] shadow-[0_18px_50px_rgba(181,44,137,0.15)] lg:-translate-y-3' : 'border-[#e3dfe8] bg-white shadow-sm'}`}>{plan.featured && <div className="absolute -top-3 left-1/2 -translate-x-1/2 rounded-full bg-[#ba2b8a] px-4 py-1.5 text-[11px] font-bold uppercase tracking-wider text-white">Most popular</div>}<h3 className="text-lg font-semibold">{plan.name}</h3><p className="mt-1 text-sm text-[#8b8694]">{plan.detail}</p><div className="mt-7 flex items-baseline gap-1"><span className="text-4xl font-semibold tracking-[-0.06em]">{plan.price}</span>{plan.price !== '$0' && <span className="text-sm text-[#8b8694]">/month</span>}</div><button className={`mt-7 w-full rounded-full py-3 text-sm font-semibold transition hover:-translate-y-0.5 ${plan.featured ? 'bg-[#ba2b8a] text-white shadow-lg shadow-[#ba2b8a]/20 hover:bg-[#a7247e]' : 'border border-[#ddd8e5] bg-white text-[#554f61] hover:border-[#ba2b8a] hover:text-[#a7247e]'}`}>{plan.featured ? 'Upgrade now' : plan.name === 'Free' ? 'Current plan' : 'Coming soon'}</button><ul className="mt-7 space-y-3 border-t border-[#eeeaf0] pt-6">{plan.features.map((item) => <li key={item} className="flex items-center gap-2 text-sm text-[#676171]"><Check size={15} className="text-[#b52c89]" /> {item}</li>)}</ul></article>)}</div></div></section>

      <footer className="border-t border-[#ebe7ef] bg-white px-5 py-8 sm:px-8 lg:px-10"><div className="mx-auto flex max-w-6xl flex-col items-center justify-between gap-4 text-center sm:flex-row sm:text-left"><p className="text-xs text-[#9993a1]">A calmer way to learn STEM.</p></div></footer>
    </main>
  )
}
