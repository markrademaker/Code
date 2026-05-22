"use client";

import { useState } from "react";

type Status = "UNPAID" | "AWAITING_VERIFICATION" | "PAID" | "REFUNDED";

export function PayBookingActions({
  bookingId,
  paymentStatus,
  hasAmount,
  stripeConfigured,
}: {
  bookingId: string;
  paymentStatus: Status;
  hasAmount: boolean;
  stripeConfigured: boolean;
}) {
  const [status, setStatus] = useState<Status>(paymentStatus);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function startOnlinePayment() {
    setBusy(true);
    setError(null);
    const res = await fetch(`/api/pay/${bookingId}/checkout`, { method: "POST" });
    const j = await res.json().catch(() => ({ error: "Unexpected response" }));
    if (!res.ok || !j.url) {
      setBusy(false);
      setError(j.error ?? "Could not start payment");
      return;
    }
    window.location.href = j.url;
  }

  async function markBankTransferSent() {
    setBusy(true);
    setError(null);
    const res = await fetch(`/api/my-bookings/${bookingId}/paid`, { method: "POST" });
    setBusy(false);
    if (!res.ok) {
      const j = await res.json().catch(() => ({}));
      setError(j.error ?? "Could not update");
      return;
    }
    setStatus("AWAITING_VERIFICATION");
  }

  if (status === "PAID") {
    return (
      <p className="mt-6 rounded-2xl bg-olive/15 px-4 py-3 text-sm text-olive">
        Payment confirmed — thank you. We&apos;re looking forward to having you.
      </p>
    );
  }

  return (
    <div className="mt-6 grid gap-3">
      {stripeConfigured && hasAmount && (
        <button
          onClick={startOnlinePayment}
          disabled={busy}
          className="flex items-center justify-center gap-2 rounded-full bg-ocean px-6 py-4 font-medium text-whitewash shadow-glow disabled:opacity-60"
        >
          {busy ? "Redirecting…" : "Pay online"}
          <span className="text-xs text-whitewash/70" aria-hidden>
            · Card · iDEAL · PayPal
          </span>
        </button>
      )}

      {status === "AWAITING_VERIFICATION" ? (
        <p className="rounded-2xl bg-sand/60 px-4 py-3 text-sm text-ink/80">
          We&apos;ve been notified that you&apos;ve transferred the amount.
          We&apos;ll confirm once it arrives.
        </p>
      ) : (
        <button
          onClick={markBankTransferSent}
          disabled={busy}
          className="rounded-full border border-ink/15 bg-white px-6 py-3 text-sm font-medium text-ink hover:bg-whitewash disabled:opacity-60"
        >
          I&apos;ve made a bank transfer instead
        </button>
      )}

      {error && (
        <p className="rounded-2xl bg-terracotta/10 px-4 py-3 text-sm text-terracotta">
          {error}
        </p>
      )}
      <p className="text-xs text-ink/55">
        {stripeConfigured
          ? "Online payments are processed by Stripe. Your card details never touch our servers."
          : "Online payment isn't set up yet. Bank transfer details are above."}
      </p>
    </div>
  );
}
