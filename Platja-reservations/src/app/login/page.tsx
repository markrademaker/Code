"use client";

import Link from "next/link";
import { useState } from "react";

type Mode = "login" | "signup";

export default function LoginPage({
  searchParams,
}: {
  searchParams: { next?: string; mode?: string };
}) {
  const [mode, setMode] = useState<Mode>(searchParams.mode === "signup" ? "signup" : "login");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function onSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const form = e.currentTarget;
    setSubmitting(true);
    setError(null);

    const formData = new FormData(form);
    const payload = Object.fromEntries(formData.entries());
    const endpoint = mode === "signup" ? "/api/auth/signup" : "/api/auth/login";

    try {
      const res = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const text = await res.text();
      let json: { error?: string } = {};
      try {
        json = text ? JSON.parse(text) : {};
      } catch {
        json = { error: `Server returned ${res.status}` };
      }
      if (!res.ok) {
        setSubmitting(false);
        setError(json.error ?? "Something went wrong");
        return;
      }
      const next = searchParams.next?.startsWith("/") ? searchParams.next : "/";
      window.location.href = next;
    } catch (err) {
      setSubmitting(false);
      setError(err instanceof Error ? err.message : "Network error");
    }
  }

  const isSignup = mode === "signup";

  return (
    <main className="relative min-h-[100dvh] overflow-hidden">
      <div
        aria-hidden
        className="absolute inset-0 -z-20 bg-cover bg-center"
        style={{
          backgroundImage:
            "linear-gradient(180deg, rgba(250,246,236,0.10) 0%, rgba(61,47,36,0.30) 100%), url('/login-bg.jpg')",
        }}
      />
      <div
        aria-hidden
        className="absolute inset-0 -z-30 bg-gradient-to-br from-sand via-sunset/40 to-ocean/60"
      />

      <div className="relative flex min-h-[100dvh] items-center justify-center px-5 py-10 sm:px-6">
        <div className="w-full max-w-md rounded-3xl bg-white/75 p-7 shadow-glow ring-1 ring-white/40 backdrop-blur-xl sm:p-9">
          <Link
            href="/"
            className="inline-block font-display text-lg font-semibold text-ink"
          >
            Villa Mas Nou
          </Link>
          <p className="mt-1 text-xs uppercase tracking-[0.2em] text-ink/55">
            Platja d&apos;Aro · Costa Brava
          </p>

          <h1 className="mt-6 font-display text-3xl font-semibold text-ink sm:text-4xl">
            {isSignup ? "Create your account" : "Welcome back"}
          </h1>
          <p className="mt-2 text-sm text-ink/70 sm:text-base">
            {isSignup
              ? "Save your details so booking takes seconds next time."
              : "Sign in to request a booking with your saved details."}
          </p>

          <form onSubmit={onSubmit} className="mt-6 grid gap-4">
            {isSignup && (
              <Field label="Your name" name="name" required autoComplete="name" />
            )}
            <Field label="Email" name="email" type="email" required autoComplete="email" />
            {isSignup && (
              <Field
                label="Phone (optional)"
                name="phone"
                type="tel"
                autoComplete="tel"
              />
            )}
            <Field
              label="Password"
              name="password"
              type="password"
              required
              autoComplete={isSignup ? "new-password" : "current-password"}
              minLength={isSignup ? 8 : undefined}
            />
            {isSignup && (
              <Field
                label="Activation code"
                name="inviteCode"
                required
                autoComplete="off"
              />
            )}
            {error && (
              <p className="rounded-xl bg-terracotta/20 px-4 py-3 text-sm text-terracotta">
                {error}
              </p>
            )}
            <button
              type="submit"
              disabled={submitting}
              className="mt-2 rounded-full bg-ocean px-6 py-3 font-medium text-whitewash shadow-glow disabled:opacity-60"
            >
              {submitting
                ? isSignup
                  ? "Creating account…"
                  : "Signing in…"
                : isSignup
                  ? "Create account"
                  : "Sign in"}
            </button>
          </form>

          <p className="mt-6 text-center text-sm text-ink/75">
            {isSignup ? (
              <>
                Already have an account?{" "}
                <button
                  type="button"
                  onClick={() => {
                    setMode("login");
                    setError(null);
                  }}
                  className="font-medium text-ocean underline"
                >
                  Sign in
                </button>
              </>
            ) : (
              <>
                New here?{" "}
                <button
                  type="button"
                  onClick={() => {
                    setMode("signup");
                    setError(null);
                  }}
                  className="font-medium text-ocean underline"
                >
                  Create an account
                </button>
              </>
            )}
          </p>
        </div>
      </div>
    </main>
  );
}

function Field({
  label,
  name,
  type = "text",
  required,
  minLength,
  autoComplete,
}: {
  label: string;
  name: string;
  type?: string;
  required?: boolean;
  minLength?: number;
  autoComplete?: string;
}) {
  return (
    <label className="block">
      <span className="text-sm font-medium text-ink">{label}</span>
      <input
        name={name}
        type={type}
        required={required}
        minLength={minLength}
        autoComplete={autoComplete}
        className="mt-1 w-full rounded-2xl border border-ink/15 bg-white/90 px-4 py-3 text-ink shadow-soft focus:border-sea focus:outline-none focus:ring-2 focus:ring-sea/40"
      />
    </label>
  );
}
