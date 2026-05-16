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
    <main className="mx-auto flex min-h-[70vh] max-w-md flex-col justify-center px-5 py-12">
      <h1 className="font-display text-2xl font-semibold sm:text-3xl">
        {isSignup ? "Create your account" : "Welcome back"}
      </h1>
      <p className="mt-2 text-deep/70">
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
        {error && (
          <p className="rounded-xl bg-terracotta/10 px-4 py-3 text-terracotta">{error}</p>
        )}
        <button
          type="submit"
          disabled={submitting}
          className="mt-2 rounded-full bg-deep px-6 py-3 font-medium text-white disabled:opacity-60"
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

      <p className="mt-6 text-center text-sm text-deep/70">
        {isSignup ? (
          <>
            Already have an account?{" "}
            <button
              type="button"
              onClick={() => {
                setMode("login");
                setError(null);
              }}
              className="font-medium text-deep underline"
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
              className="font-medium text-deep underline"
            >
              Create an account
            </button>
          </>
        )}
      </p>
      <p className="mt-2 text-center text-xs text-deep/50">
        <Link href="/" className="hover:underline">
          ← Back to villa
        </Link>
      </p>
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
      <span className="text-sm font-medium text-deep">{label}</span>
      <input
        name={name}
        type={type}
        required={required}
        minLength={minLength}
        autoComplete={autoComplete}
        className="mt-1 w-full rounded-xl border border-deep/15 bg-white px-4 py-3 shadow-sm focus:border-sea focus:outline-none focus:ring-2 focus:ring-sea/30"
      />
    </label>
  );
}
