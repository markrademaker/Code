"use client";

import Link from "next/link";
import { useState } from "react";

type Status =
  | { kind: "idle" }
  | { kind: "submitting" }
  | { kind: "success" }
  | { kind: "error"; message: string };

export type BookingFormUser = {
  name: string;
  email: string;
  phone: string | null;
};

export function BookingForm({ user }: { user: BookingFormUser | null }) {
  const [status, setStatus] = useState<Status>({ kind: "idle" });

  if (!user) {
    return (
      <section id="book" className="mx-auto max-w-3xl px-5 py-16 sm:px-6 sm:py-24">
        <div className="rounded-3xl bg-white p-8 shadow-soft ring-1 ring-ink/5 sm:p-10">
          <p className="text-xs uppercase tracking-[0.2em] text-ink/55">Request a booking</p>
          <h2 className="mt-2 font-display text-3xl font-semibold sm:text-4xl">
            Sign in to book in seconds
          </h2>
          <p className="mt-3 text-base text-ink/70 sm:text-lg">
            We&apos;ll save your name, email and phone so you only pick the
            dates next time.
          </p>
          <div className="mt-7 flex flex-col gap-3 sm:flex-row">
            <Link
              href="/login?mode=signup&next=%2F%23book"
              className="rounded-full bg-ocean px-7 py-3.5 text-center font-medium text-whitewash shadow-glow hover:bg-ocean/90"
            >
              Create account
            </Link>
            <Link
              href="/login?next=%2F%23book"
              className="rounded-full border border-ink/15 px-7 py-3.5 text-center font-medium text-ink hover:bg-whitewash"
            >
              Sign in
            </Link>
          </div>
        </div>
      </section>
    );
  }

  async function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const form = e.currentTarget;
    setStatus({ kind: "submitting" });
    const formData = new FormData(form);
    const payload = Object.fromEntries(formData.entries());

    try {
      const res = await fetch("/api/book", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const text = await res.text();
      let json: { error?: string; ok?: boolean; warning?: string } = {};
      try {
        json = text ? JSON.parse(text) : {};
      } catch {
        json = { error: text.slice(0, 200) || `Server returned ${res.status}` };
      }
      if (!res.ok) {
        setStatus({
          kind: "error",
          message: json.error ?? `Server error (${res.status})`,
        });
        return;
      }
      setStatus({ kind: "success" });
      form.reset();
    } catch (err) {
      const message = err instanceof Error ? err.message : "Network error";
      setStatus({ kind: "error", message });
    }
  }

  const submitting = status.kind === "submitting";

  return (
    <section id="book" className="mx-auto max-w-3xl px-5 py-16 sm:px-6 sm:py-24">
      <p className="text-xs uppercase tracking-[0.2em] text-ink/55">Booking</p>
      <h2 className="mt-2 font-display text-3xl font-semibold sm:text-4xl">
        Request your dates
      </h2>
      <p className="mt-3 text-base text-ink/70 sm:text-lg">
        Pick check-in and check-out and we&apos;ll email you back to confirm.
      </p>

      <div className="mt-7 flex flex-wrap items-center justify-between gap-3 rounded-2xl bg-white p-4 text-sm shadow-soft ring-1 ring-ink/5 sm:p-5">
        <div className="text-ink/80">
          Booking as <strong className="text-ink">{user.name}</strong>{" "}
          <span className="text-ink/55">({user.email})</span>
        </div>
        <Link
          href="/my-bookings"
          className="text-xs font-medium text-ocean hover:underline"
        >
          My bookings →
        </Link>
      </div>

      <form onSubmit={handleSubmit} className="mt-6 grid gap-5">
        <div className="grid gap-5 sm:grid-cols-2">
          <Field label="Check-in" name="checkIn" type="date" required />
          <Field label="Check-out" name="checkOut" type="date" required />
          <Field
            label="Number of guests"
            name="guests"
            type="number"
            min={1}
            max={20}
            defaultValue={2}
            required
          />
        </div>
        <label className="block">
          <span className="text-sm font-medium text-ink">Message (optional)</span>
          <textarea
            name="message"
            rows={4}
            className="mt-1 w-full rounded-2xl border border-ink/15 bg-white px-4 py-3 text-ink shadow-soft focus:border-sea focus:outline-none focus:ring-2 focus:ring-sea/30"
            placeholder="Anything we should know — kids, dietary, arrival time…"
          />
        </label>

        <button
          type="submit"
          disabled={submitting}
          className="mt-2 w-full rounded-full bg-ocean px-8 py-4 font-medium text-whitewash shadow-glow transition hover:bg-ocean/90 disabled:opacity-60 sm:w-auto sm:self-start sm:py-3.5"
        >
          {submitting ? "Sending…" : "Send booking request"}
        </button>

        {status.kind === "success" && (
          <p className="rounded-2xl bg-sea/15 px-4 py-3 text-ocean">
            Thanks! Your request was sent. We&apos;ll reply by email shortly.
          </p>
        )}
        {status.kind === "error" && (
          <p className="rounded-2xl bg-terracotta/10 px-4 py-3 text-terracotta">
            {status.message}
          </p>
        )}
      </form>
    </section>
  );
}

function Field({
  label,
  name,
  type = "text",
  required,
  min,
  max,
  defaultValue,
}: {
  label: string;
  name: string;
  type?: string;
  required?: boolean;
  min?: number;
  max?: number;
  defaultValue?: string | number;
}) {
  return (
    <label className="block">
      <span className="text-sm font-medium text-ink">{label}</span>
      <input
        name={name}
        type={type}
        required={required}
        min={min}
        max={max}
        defaultValue={defaultValue}
        className="mt-1 w-full rounded-2xl border border-ink/15 bg-white px-4 py-3 text-ink shadow-soft focus:border-sea focus:outline-none focus:ring-2 focus:ring-sea/30"
      />
    </label>
  );
}
