"use client";

import { useState } from "react";

export default function LoginPage({
  searchParams,
}: {
  searchParams: { next?: string; error?: string };
}) {
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(
    searchParams.error === "1" ? "Wrong password" : null,
  );

  async function onSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    const formData = new FormData(e.currentTarget);
    const res = await fetch("/api/admin/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        password: formData.get("password"),
        next: searchParams.next ?? "/admin",
      }),
    });
    if (res.ok) {
      const { redirect } = await res.json();
      window.location.href = redirect ?? "/admin";
      return;
    }
    setSubmitting(false);
    setError("Wrong password");
  }

  return (
    <main className="mx-auto flex min-h-[60vh] max-w-md flex-col justify-center px-5 py-16">
      <h1 className="font-display text-2xl font-semibold sm:text-3xl">Admin sign in</h1>
      <p className="mt-2 text-deep/70">Enter the admin password to manage bookings.</p>
      <form onSubmit={onSubmit} className="mt-6 grid gap-4">
        <label className="block">
          <span className="text-sm font-medium text-deep">Password</span>
          <input
            name="password"
            type="password"
            required
            autoFocus
            className="mt-1 w-full rounded-xl border border-deep/15 bg-white px-4 py-3 shadow-sm focus:border-sea focus:outline-none focus:ring-2 focus:ring-sea/30"
          />
        </label>
        {error && <p className="rounded-xl bg-terracotta/10 px-4 py-3 text-terracotta">{error}</p>}
        <button
          type="submit"
          disabled={submitting}
          className="rounded-full bg-deep px-6 py-3 font-medium text-white disabled:opacity-60"
        >
          {submitting ? "Signing in…" : "Sign in"}
        </button>
      </form>
    </main>
  );
}
