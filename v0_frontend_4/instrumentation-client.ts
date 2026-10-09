import * as Sentry from "@sentry/nextjs";

Sentry.init({
  dsn: process.env.NEXT_PUBLIC_SENTRY_DSN,

  tracesSampleRate: process.env.NODE_ENV === "development" ? 1.0 : 0.1,

  replaysSessionSampleRate: 0.1,
  replaysOnErrorSampleRate: 1.0,

  enableLogs: true,

  integrations: [Sentry.replayIntegration()],

  tracePropagationTargets: [
    "localhost",
    ...(process.env.NEXT_PUBLIC_API_BASE_URL
      ? [process.env.NEXT_PUBLIC_API_BASE_URL.replace(/\/+$/, "")]
      : []),
  ],
});

export const onRouterTransitionStart = Sentry.captureRouterTransitionStart;
