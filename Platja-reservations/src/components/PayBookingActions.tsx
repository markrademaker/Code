"use client";

import { useState } from "react";

type Status = "UNPAID" | "AWAITING_VERIFICATION" | "PAID" | "REFUNDED";

export function PayBookingActions({
  bookingId,
  paymentStatus,
}: {
  bookingId: string;
  paymentStatus: Status;
}) {
  const [status, setStatus] = useState<Status>(paymentStatus);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function markPaid() {
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
  if (status === "AWAITING_VERIFICATION") {
    return (
      <p className="mt-6 rounded-2xl bg-sand/60 px-4 py-3 text-sm text-ink/80">
        We&apos;ve been notified that you&apos;ve transferred the amount.
        We&apos;ll confirm once it arrives.
      </p>
    );
  }
  return (
    <>
      <button
        onClick={markPaid}
        disabled={busy}
        className="mt-6 rounded-full bg-ocean px-6 py-3 font-medium text-whitewash shadow-glow disabled:opacity-60"
      >
        {busy ? "Updating…" : "I've made the transfer"}
      </button>
      {error && (
        <p className="mt-3 rounded-2xl bg-terracotta/10 px-4 py-3 text-sm text-terracotta">
          {error}
        </p>
      )}
      <p className="mt-3 text-xs text-ink/55">
        Clicking this only marks the booking as awaiting verification — the
        owners will check the transfer and confirm.
      </p>
    </>
  );
}
