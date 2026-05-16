"use client";

import { useState } from "react";

type Status =
  | { kind: "idle" }
  | { kind: "submitting" }
  | { kind: "success" }
  | { kind: "error"; message: string };

export function BookingForm() {
  const [status, setStatus] = useState<Status>({ kind: "idle" });

  async function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setStatus({ kind: "submitting" });
    const formData = new FormData(e.currentTarget);
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
      e.currentTarget.reset();
    } catch (err) {
      const message = err instanceof Error ? err.message : "Network error";
      setStatus({ kind: "error", message });
    }
  }

  const submitting = status.kind === "submitting";

  return (
    <section id="book" className="mx-auto max-w-3xl px-5 py-14 sm:px-6 sm:py-20">
      <h2 className="font-display text-2xl font-semibold sm:text-3xl lg:text-4xl">
        Request a booking
      </h2>
      <p className="mt-3 text-base text-deep/70 sm:text-lg">
        Fill in your dates and we&apos;ll email you back to confirm. Your
        request is sent directly to the owners.
      </p>

      <form onSubmit={handleSubmit} className="mt-8 grid gap-5 sm:mt-10">
        <div className="grid gap-5 sm:grid-cols-2">
          <Field label="Your name" name="name" required />
          <Field label="Email" name="email" type="email" required />
          <Field label="Phone (optional)" name="phone" type="tel" />
          <Field label="Number of guests" name="guests" type="number" min={1} max={20} defaultValue={2} required />
          <Field label="Check-in" name="checkIn" type="date" required />
          <Field label="Check-out" name="checkOut" type="date" required />
        </div>
        <label className="block">
          <span className="text-sm font-medium text-deep">Message (optional)</span>
          <textarea
            name="message"
            rows={4}
            className="mt-1 w-full rounded-xl border border-deep/15 bg-white px-4 py-3 text-deep shadow-sm focus:border-sea focus:outline-none focus:ring-2 focus:ring-sea/30"
            placeholder="Anything we should know — kids, dietary, arrival time…"
          />
        </label>

        <button
          type="submit"
          disabled={submitting}
          className="mt-2 w-full rounded-full bg-deep px-8 py-4 font-medium text-white shadow-lg transition hover:bg-deep/90 disabled:opacity-60 sm:w-auto sm:self-start sm:py-3"
        >
          {submitting ? "Sending…" : "Send booking request"}
        </button>

        {status.kind === "success" && (
          <p className="rounded-xl bg-sea/10 px-4 py-3 text-sea">
            Thanks! Your request was sent. We&apos;ll reply by email shortly.
          </p>
        )}
        {status.kind === "error" && (
          <p className="rounded-xl bg-terracotta/10 px-4 py-3 text-terracotta">
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
      <span className="text-sm font-medium text-deep">{label}</span>
      <input
        name={name}
        type={type}
        required={required}
        min={min}
        max={max}
        defaultValue={defaultValue}
        className="mt-1 w-full rounded-xl border border-deep/15 bg-white px-4 py-3 text-deep shadow-sm focus:border-sea focus:outline-none focus:ring-2 focus:ring-sea/30"
      />
    </label>
  );
}
