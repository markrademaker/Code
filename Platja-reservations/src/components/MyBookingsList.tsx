"use client";

import { useState } from "react";
import { format, parseISO } from "date-fns";
import { MessageThread } from "@/components/MessageThread";

type Booking = {
  id: string;
  checkIn: string;
  checkOut: string;
  guests: number;
  message: string | null;
  status: "PENDING" | "TENTATIVE" | "CONFIRMED" | "DECLINED" | "CANCELLED";
  ownerNote: string | null;
  createdAt: string;
};

const STATUS_STYLES: Record<Booking["status"], string> = {
  PENDING: "bg-sunset/20 text-terracotta",
  TENTATIVE: "bg-sand text-ink",
  CONFIRMED: "bg-sea/15 text-ocean",
  DECLINED: "bg-ink/10 text-ink/70",
  CANCELLED: "bg-ink/10 text-ink/70",
};

const STATUS_LABEL: Record<Booking["status"], string> = {
  PENDING: "Waiting for response",
  TENTATIVE: "Tentatively held",
  CONFIRMED: "Confirmed",
  DECLINED: "Declined",
  CANCELLED: "Cancelled",
};

export function MyBookingsList({ bookings: initial }: { bookings: Booking[] }) {
  const [bookings, setBookings] = useState(initial);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [openId, setOpenId] = useState<string | null>(null);

  async function cancel(b: Booking) {
    if (!confirm(`Cancel your booking from ${b.checkIn} to ${b.checkOut}?`)) return;
    setBusyId(b.id);
    setError(null);
    const res = await fetch(`/api/my-bookings/${b.id}`, { method: "POST" });
    setBusyId(null);
    if (!res.ok) {
      const j = await res.json().catch(() => ({}));
      setError(j.error ?? "Could not cancel");
      return;
    }
    setBookings((bs) =>
      bs.map((x) => (x.id === b.id ? { ...x, status: "CANCELLED" } : x)),
    );
  }

  if (bookings.length === 0) {
    return (
      <div className="rounded-3xl bg-white p-8 text-center shadow-soft ring-1 ring-ink/5">
        <p className="text-ink/70">No bookings yet.</p>
      </div>
    );
  }

  return (
    <div className="grid gap-4">
      {error && (
        <p className="rounded-2xl bg-terracotta/10 px-4 py-3 text-terracotta">{error}</p>
      )}
      {bookings.map((b) => {
        const closed = b.status === "CANCELLED" || b.status === "DECLINED";
        const showOwnerNote =
          b.ownerNote &&
          (b.status === "CONFIRMED" || b.status === "TENTATIVE");
        const isOpen = openId === b.id;
        return (
          <div
            key={b.id}
            className="rounded-3xl bg-white p-5 shadow-soft ring-1 ring-ink/5"
          >
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <p className="font-display text-lg font-semibold text-ink">
                  {format(parseISO(b.checkIn), "EEE d MMM yyyy")} →{" "}
                  {format(parseISO(b.checkOut), "EEE d MMM yyyy")}
                </p>
                <p className="mt-1 text-sm text-ink/60">
                  {b.guests} {b.guests === 1 ? "guest" : "guests"} · requested{" "}
                  {format(parseISO(b.createdAt), "d MMM yyyy")}
                </p>
              </div>
              <span
                className={`rounded-full px-3 py-1 text-xs font-medium ${STATUS_STYLES[b.status]}`}
              >
                {STATUS_LABEL[b.status]}
              </span>
            </div>

            {showOwnerNote && (
              <div className="mt-4 rounded-2xl bg-sea/10 p-4">
                <p className="text-xs font-medium uppercase tracking-wider text-ocean">
                  Note from the owners
                </p>
                <p className="mt-2 whitespace-pre-wrap text-sm leading-relaxed text-ink">
                  {b.ownerNote}
                </p>
              </div>
            )}

            {b.message && (
              <p className="mt-3 whitespace-pre-wrap rounded-2xl bg-whitewash px-4 py-3 text-sm text-ink/75">
                {b.message}
              </p>
            )}

            <div className="mt-4 flex flex-wrap justify-between gap-2">
              <button
                onClick={() => setOpenId(isOpen ? null : b.id)}
                className="rounded-full bg-sand/60 px-4 py-2 text-sm font-medium text-ink hover:bg-sand"
              >
                {isOpen ? "Hide messages" : "Messages"}
              </button>
              {!closed && (
                <button
                  onClick={() => cancel(b)}
                  disabled={busyId === b.id}
                  className="rounded-full px-4 py-2 text-sm font-medium text-ink/60 hover:bg-ink/5 disabled:opacity-50"
                >
                  {busyId === b.id ? "Cancelling…" : "Cancel booking"}
                </button>
              )}
            </div>

            {isOpen && (
              <div className="mt-4">
                <MessageThread bookingId={b.id} side="guest" />
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
