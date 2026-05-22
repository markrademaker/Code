"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { formatEuro } from "@/lib/pricing";
import { SectionMark, SunMark } from "@/components/Marks";

type Status =
  | { kind: "idle" }
  | { kind: "submitting" }
  | { kind: "success" }
  | { kind: "error"; message: string };

type QuoteData = {
  nights: number;
  coveredNights: number;
  missingNights: number;
  totalCents: number | null;
};

const DATE_RE = /^\d{4}-\d{2}-\d{2}$/;

export type BookingFormUser = {
  name: string;
  email: string;
  phone: string | null;
};

export function BookingForm({ user }: { user: BookingFormUser | null }) {
  const [status, setStatus] = useState<Status>({ kind: "idle" });
  const [checkIn, setCheckIn] = useState("");
  const [checkOut, setCheckOut] = useState("");
  const [quote, setQuote] = useState<QuoteData | null>(null);
  const [quoteLoading, setQuoteLoading] = useState(false);

  useEffect(() => {
    if (!DATE_RE.test(checkIn) || !DATE_RE.test(checkOut) || checkOut <= checkIn) {
      setQuote(null);
      return;
    }
    let alive = true;
    setQuoteLoading(true);
    fetch(`/api/quote?checkIn=${checkIn}&checkOut=${checkOut}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((j) => {
        if (!alive) return;
        setQuote(j);
        setQuoteLoading(false);
      })
      .catch(() => {
        if (!alive) return;
        setQuoteLoading(false);
      });
    return () => {
      alive = false;
    };
  }, [checkIn, checkOut]);

  if (!user) {
    return (
      <section id="book" className="mx-auto max-w-7xl px-5 py-20 sm:px-8 sm:py-28">
        <div className="grid gap-10 rounded-[2rem] bg-white p-8 shadow-soft ring-1 ring-ink/5 sm:p-12 lg:grid-cols-[1fr_1fr] lg:gap-16">
          <div>
            <SectionMark number="III" label="Reserve" />
            <h2 className="mt-4 font-display text-4xl font-semibold leading-tight sm:text-5xl">
              Sign in to book
              <br />
              <span className="italic text-terracotta">in seconds</span>.
            </h2>
            <p className="mt-5 text-base text-ink/70 sm:text-lg">
              We save your name, email, and phone — so next time you only
              pick the dates.
            </p>
          </div>
          <div className="flex flex-col justify-center gap-3 lg:items-start">
            <Link
              href="/login?mode=signup&next=%2F%23book"
              className="w-full rounded-full bg-ocean px-7 py-3.5 text-center font-medium text-whitewash shadow-glow hover:bg-ocean/90 sm:w-auto"
            >
              Create account
            </Link>
            <Link
              href="/login?next=%2F%23book"
              className="w-full rounded-full border border-ink/15 px-7 py-3.5 text-center font-medium text-ink hover:bg-whitewash sm:w-auto"
            >
              Sign in
            </Link>
            <p className="mt-3 flex items-center gap-2 text-xs text-ink/55">
              <SunMark className="h-3 w-3 text-terracotta" />
              Activation code required for sign-up
            </p>
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
    <section id="book" className="mx-auto max-w-7xl px-5 py-20 sm:px-8 sm:py-28">
      <div className="grid gap-10 lg:grid-cols-[1fr_1.3fr] lg:gap-20">
        <aside className="lg:sticky lg:top-28 lg:self-start">
          <SectionMark number="III" label="Reserve" />
          <h2 className="mt-4 font-display text-4xl font-semibold leading-tight sm:text-5xl lg:text-6xl">
            Request <span className="italic text-terracotta">your</span>
            <br />
            dates.
          </h2>
          <p className="mt-5 max-w-md text-base text-ink/70 sm:text-lg">
            Pick check-in and check-out — we&apos;ll email you back to
            confirm, usually within a day.
          </p>

          <ul className="mt-8 grid gap-3 text-sm text-ink/75">
            {["Saturday-to-Saturday is easiest", "Minimum stay one week in summer", "Payment due seven days before arrival"].map(
              (line) => (
                <li key={line} className="flex items-start gap-3">
                  <SunMark className="mt-1 h-3 w-3 shrink-0 text-terracotta" />
                  <span>{line}</span>
                </li>
              ),
            )}
          </ul>

          <div className="mt-8 flex items-center gap-3 rounded-2xl bg-white p-4 text-sm shadow-soft ring-1 ring-ink/5">
            <div className="flex-1 text-ink/80">
              Booking as <strong className="text-ink">{user.name}</strong>
              <br />
              <span className="text-xs text-ink/55">{user.email}</span>
            </div>
            <Link
              href="/my-bookings"
              className="text-xs font-medium text-ocean hover:underline"
            >
              My bookings →
            </Link>
          </div>
        </aside>

        <div>

      <form onSubmit={handleSubmit} className="mt-6 grid gap-5">
        <div className="grid gap-5 sm:grid-cols-2">
          <Field
            label="Check-in"
            name="checkIn"
            type="date"
            required
            value={checkIn}
            onChange={setCheckIn}
          />
          <Field
            label="Check-out"
            name="checkOut"
            type="date"
            required
            value={checkOut}
            onChange={setCheckOut}
          />
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

        {quote && quote.nights > 0 && (
          <div className="rounded-2xl bg-sand/40 p-4 sm:p-5">
            <p className="text-xs uppercase tracking-wider text-ink/55">
              Estimated price
            </p>
            <p className="mt-1 font-display text-2xl font-semibold text-ink">
              {quote.totalCents != null
                ? formatEuro(quote.totalCents)
                : "Price on request"}
              <span className="ml-2 text-sm font-normal text-ink/55">
                · {quote.nights} {quote.nights === 1 ? "night" : "nights"}
              </span>
            </p>
            {quote.missingNights > 0 && (
              <p className="mt-1 text-xs text-ink/60">
                {quote.missingNights}{" "}
                {quote.missingNights === 1 ? "night" : "nights"} not yet
                priced — we&apos;ll confirm by email.
              </p>
            )}
            <p className="mt-2 text-xs text-ink/60">
              Full payment due 7 days before check-in.
            </p>
          </div>
        )}
        {!quote && quoteLoading && (
          <p className="text-xs text-ink/55">Calculating price…</p>
        )}
        <label className="block">
          <span className="text-sm font-medium text-ink">
            Names of the other guests
          </span>
          <span className="block text-xs text-ink/55">
            One name per line. You don&apos;t need to add yourself.
          </span>
          <textarea
            name="guestNames"
            rows={3}
            className="mt-1 w-full rounded-2xl border border-ink/15 bg-white px-4 py-3 text-ink shadow-soft focus:border-sea focus:outline-none focus:ring-2 focus:ring-sea/30"
            placeholder={"Anna van Dijk\nDiego Ferrer\n…"}
          />
        </label>
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
        </div>
      </div>
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
  value,
  onChange,
}: {
  label: string;
  name: string;
  type?: string;
  required?: boolean;
  min?: number;
  max?: number;
  defaultValue?: string | number;
  value?: string;
  onChange?: (v: string) => void;
}) {
  const controlled = value !== undefined && onChange !== undefined;
  return (
    <label className="block">
      <span className="text-sm font-medium text-ink">{label}</span>
      <input
        name={name}
        type={type}
        required={required}
        min={min}
        max={max}
        {...(controlled
          ? { value, onChange: (e: React.ChangeEvent<HTMLInputElement>) => onChange?.(e.target.value) }
          : { defaultValue })}
        className="mt-1 w-full rounded-2xl border border-ink/15 bg-white px-4 py-3 text-ink shadow-soft focus:border-sea focus:outline-none focus:ring-2 focus:ring-sea/30"
      />
    </label>
  );
}
