"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { format, parseISO } from "date-fns";
import { AdminPlanningCalendar } from "@/components/AdminPlanningCalendar";
import { MessageThread } from "@/components/MessageThread";
import { ChatWidget } from "@/components/ChatWidget";

export type AdminBooking = {
  id: string;
  name: string;
  email: string;
  phone: string | null;
  guests: number;
  guestNames: string | null;
  checkIn: string;
  checkOut: string;
  message: string | null;
  status: "PENDING" | "TENTATIVE" | "CONFIRMED" | "DECLINED" | "CANCELLED";
  notes: string | null;
  ownerNote: string | null;
  totalAmountCents: number | null;
  paymentStatus: "UNPAID" | "AWAITING_VERIFICATION" | "PAID" | "REFUNDED";
  paymentDueDate: string | null;
  createdAt: string;
};

const PAY_STYLES: Record<AdminBooking["paymentStatus"], string> = {
  UNPAID: "bg-sunset/20 text-terracotta",
  AWAITING_VERIFICATION: "bg-sand text-ink",
  PAID: "bg-olive/20 text-olive",
  REFUNDED: "bg-ink/10 text-ink/70",
};

function euro(cents: number | null): string | null {
  if (cents == null) return null;
  return new Intl.NumberFormat("nl-NL", {
    style: "currency",
    currency: "EUR",
    minimumFractionDigits: 0,
  }).format(cents / 100);
}

const STATUS_STYLES: Record<AdminBooking["status"], string> = {
  PENDING:   "bg-yellow-100 text-yellow-900",
  TENTATIVE: "bg-amber-200 text-amber-900",
  CONFIRMED: "bg-emerald-200 text-emerald-900",
  DECLINED:  "bg-red-100 text-red-900",
  CANCELLED: "bg-gray-200 text-gray-700",
};

const STATUS_OPTIONS: AdminBooking["status"][] = [
  "PENDING",
  "TENTATIVE",
  "CONFIRMED",
  "DECLINED",
  "CANCELLED",
];

const FILTERS = ["All", "Active", "Pending", "Past"] as const;
type Filter = (typeof FILTERS)[number];

export function AdminDashboard({ bookings: initial }: { bookings: AdminBooking[] }) {
  const [bookings, setBookings] = useState(initial);
  const [filter, setFilter] = useState<Filter>("Active");
  const [editingId, setEditingId] = useState<string | null>(null);
  const [openChatId, setOpenChatId] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [highlightId, setHighlightId] = useState<string | null>(null);

  function jumpToBooking(id: string) {
    const el = document.getElementById(`booking-${id}`);
    if (!el) return;
    el.scrollIntoView({ behavior: "smooth", block: "center" });
    setHighlightId(id);
    setTimeout(() => setHighlightId((h) => (h === id ? null : h)), 2000);
  }

  const today = new Date().toISOString().slice(0, 10);

  const filtered = useMemo(() => {
    return bookings
      .filter((b) => {
        if (filter === "All") return true;
        if (filter === "Pending") return b.status === "PENDING";
        if (filter === "Past") return b.checkOut < today;
        // Active
        return (
          b.checkOut >= today &&
          (b.status === "CONFIRMED" || b.status === "TENTATIVE" || b.status === "PENDING")
        );
      })
      .sort((a, b) => a.checkIn.localeCompare(b.checkIn));
  }, [bookings, filter, today]);

  async function patch(id: string, body: Record<string, unknown>): Promise<boolean> {
    setBusyId(id);
    setError(null);
    const res = await fetch(`/api/admin/bookings/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    setBusyId(null);
    if (!res.ok) {
      const j = await res.json().catch(() => ({}));
      setError(j.error ?? "Update failed");
      return false;
    }
    return true;
  }

  async function refreshAndReplace(id: string, update: Partial<AdminBooking>): Promise<void> {
    setBookings((bs) => bs.map((b) => (b.id === id ? { ...b, ...update } : b)));
  }

  async function setStatus(b: AdminBooking, status: AdminBooking["status"]): Promise<void> {
    if (status === b.status) return;
    const ok = await patch(b.id, { status });
    if (ok) refreshAndReplace(b.id, { status });
  }

  async function saveEdit(
    b: AdminBooking,
    changes: {
      checkIn: string;
      checkOut: string;
      status: AdminBooking["status"];
      notes: string;
      ownerNote: string;
    },
  ): Promise<void> {
    const ok = await patch(b.id, changes);
    if (ok) {
      refreshAndReplace(b.id, changes);
      setEditingId(null);
    }
  }

  async function setPaymentStatus(
    b: AdminBooking,
    paymentStatus: AdminBooking["paymentStatus"],
  ): Promise<void> {
    setBusyId(b.id);
    setError(null);
    const res = await fetch(`/api/admin/bookings/${b.id}/payment`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ paymentStatus }),
    });
    setBusyId(null);
    if (!res.ok) {
      const j = await res.json().catch(() => ({}));
      setError(j.error ?? "Could not update payment");
      return;
    }
    setBookings((bs) =>
      bs.map((x) => (x.id === b.id ? { ...x, paymentStatus } : x)),
    );
  }

  async function remove(b: AdminBooking): Promise<void> {
    if (!confirm(`Delete booking from ${b.name}? This cannot be undone.`)) return;
    setBusyId(b.id);
    const res = await fetch(`/api/admin/bookings/${b.id}`, { method: "DELETE" });
    setBusyId(null);
    if (res.ok) setBookings((bs) => bs.filter((x) => x.id !== b.id));
  }

  async function logout(): Promise<void> {
    await fetch("/api/auth/logout", { method: "POST" });
    window.location.href = "/";
  }

  return (
    <main className="mx-auto max-w-6xl px-5 py-8 sm:px-6 sm:py-12">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className="text-xs uppercase tracking-[0.25em] text-ink/60">Admin</p>
          <h1 className="mt-1 font-display text-3xl font-semibold sm:text-4xl">Planning</h1>
        </div>
        <div className="flex gap-2">
          <Link
            href="/admin/rates"
            className="rounded-full bg-white px-4 py-2 text-sm font-medium shadow-soft ring-1 ring-ink/10 hover:bg-ink/5"
          >
            Rates
          </Link>
          <Link
            href="/admin/restaurants"
            className="rounded-full bg-white px-4 py-2 text-sm font-medium shadow-soft ring-1 ring-ink/10 hover:bg-ink/5"
          >
            Restaurants
          </Link>
          <button
            onClick={logout}
            className="rounded-full bg-white px-4 py-2 text-sm font-medium shadow-soft ring-1 ring-ink/10 hover:bg-ink/5"
          >
            Sign out
          </button>
        </div>
      </div>

      <AdminPlanningCalendar bookings={bookings} onSelect={jumpToBooking} />

      <section className="mt-10">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h2 className="font-display text-xl font-semibold sm:text-2xl">Bookings</h2>
          <div className="flex gap-1 rounded-full bg-white p-1 shadow-sm ring-1 ring-deep/10">
            {FILTERS.map((f) => (
              <button
                key={f}
                onClick={() => setFilter(f)}
                className={`rounded-full px-3 py-1.5 text-sm ${
                  filter === f ? "bg-deep text-white" : "text-deep hover:bg-deep/5"
                }`}
              >
                {f}
              </button>
            ))}
          </div>
        </div>

        {error && (
          <p className="mt-4 rounded-xl bg-terracotta/10 px-4 py-3 text-terracotta">{error}</p>
        )}

        <ul className="mt-4 grid gap-3">
          {filtered.length === 0 && (
            <li className="rounded-2xl bg-white p-6 text-deep/60 shadow-sm ring-1 ring-deep/5">
              No bookings.
            </li>
          )}
          {filtered.map((b) => {
            const editing = editingId === b.id;
            const isHi = highlightId === b.id;
            const names = (b.guestNames ?? "")
              .split(/\r?\n|,/)
              .map((n) => n.trim())
              .filter(Boolean);
            return (
              <li
                key={b.id}
                id={`booking-${b.id}`}
                className={`scroll-mt-24 rounded-2xl bg-white p-5 shadow-soft ring-1 transition ${
                  isHi ? "ring-terracotta ring-2" : "ring-ink/5"
                }`}
              >
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <p className="font-semibold text-ink">{b.name}</p>
                    <p className="text-sm text-ink/70">
                      <a href={`mailto:${b.email}`} className="hover:underline">
                        {b.email}
                      </a>
                      {b.phone && <> · {b.phone}</>}
                    </p>
                    <p className="mt-1 text-sm text-ink/80">
                      {format(parseISO(b.checkIn), "EEE d MMM yyyy")}{" → "}
                      {format(parseISO(b.checkOut), "EEE d MMM yyyy")}
                      <span className="ml-2 text-ink/60">· {b.guests} guests</span>
                    </p>
                  </div>
                  <span
                    className={`rounded-full px-3 py-1 text-xs font-medium ${STATUS_STYLES[b.status]}`}
                  >
                    {b.status}
                  </span>
                </div>

                <div className="mt-3 flex flex-wrap items-center gap-3 rounded-xl bg-whitewash px-4 py-3">
                  <div className="flex-1">
                    <p className="text-[10px] font-semibold uppercase tracking-wider text-ink/55">
                      Payment
                    </p>
                    <p className="mt-0.5 font-display text-lg font-semibold text-ink">
                      {euro(b.totalAmountCents) ?? "Price on request"}
                      {b.paymentDueDate && b.paymentStatus !== "PAID" && (
                        <span className="ml-2 text-xs font-normal text-ink/55">
                          due {format(parseISO(b.paymentDueDate), "d MMM")}
                        </span>
                      )}
                    </p>
                  </div>
                  <span
                    className={`rounded-full px-3 py-1 text-xs font-medium ${PAY_STYLES[b.paymentStatus]}`}
                  >
                    {b.paymentStatus.replace("_", " ").toLowerCase()}
                  </span>
                  {b.paymentStatus !== "PAID" && (
                    <button
                      onClick={() => setPaymentStatus(b, "PAID")}
                      disabled={busyId === b.id}
                      className="rounded-full bg-olive px-3 py-1.5 text-xs font-medium text-whitewash hover:bg-olive/90 disabled:opacity-50"
                    >
                      Mark paid
                    </button>
                  )}
                  {b.paymentStatus === "PAID" && (
                    <button
                      onClick={() => setPaymentStatus(b, "UNPAID")}
                      disabled={busyId === b.id}
                      className="rounded-full px-3 py-1.5 text-xs font-medium text-ink/60 hover:bg-ink/5 disabled:opacity-50"
                    >
                      Unmark
                    </button>
                  )}
                </div>

                {names.length > 0 && (
                  <div className="mt-3 rounded-xl bg-sand/50 px-4 py-3">
                    <p className="text-[10px] font-semibold uppercase tracking-wider text-ink/55">
                      Other guests
                    </p>
                    <ul className="mt-1 flex flex-wrap gap-1.5">
                      {names.map((n) => (
                        <li
                          key={n}
                          className="rounded-full bg-white px-2.5 py-0.5 text-sm text-ink shadow-soft ring-1 ring-ink/5"
                        >
                          {n}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {b.message && (
                  <p className="mt-3 whitespace-pre-wrap rounded-xl bg-sand/60 px-4 py-3 text-sm text-ink/80">
                    {b.message}
                  </p>
                )}

                {b.ownerNote && (
                  <p className="mt-3 whitespace-pre-wrap rounded-xl bg-sea/10 px-4 py-3 text-sm text-ocean">
                    <span className="text-[10px] font-semibold uppercase tracking-wider">Note to guest</span>
                    <br />
                    {b.ownerNote}
                  </p>
                )}

                {editing ? (
                  <EditForm
                    booking={b}
                    busy={busyId === b.id}
                    onCancel={() => setEditingId(null)}
                    onSave={(changes) => saveEdit(b, changes)}
                  />
                ) : (
                  <div className="mt-4 flex flex-wrap gap-2">
                    {b.status === "PENDING" || b.status === "TENTATIVE" ? (
                      <>
                        <ActionBtn onClick={() => setStatus(b, "CONFIRMED")} variant="primary" disabled={busyId === b.id}>
                          Accept
                        </ActionBtn>
                        <ActionBtn onClick={() => setStatus(b, "TENTATIVE")} disabled={busyId === b.id}>
                          Tentative
                        </ActionBtn>
                        <ActionBtn onClick={() => setStatus(b, "DECLINED")} variant="danger" disabled={busyId === b.id}>
                          Decline
                        </ActionBtn>
                      </>
                    ) : null}
                    {b.status === "CONFIRMED" && (
                      <ActionBtn onClick={() => setStatus(b, "CANCELLED")} variant="danger" disabled={busyId === b.id}>
                        Cancel
                      </ActionBtn>
                    )}
                    <ActionBtn onClick={() => setEditingId(b.id)} disabled={busyId === b.id}>
                      Edit / note
                    </ActionBtn>
                    <ActionBtn onClick={() => setOpenChatId(openChatId === b.id ? null : b.id)} disabled={busyId === b.id}>
                      {openChatId === b.id ? "Hide messages" : "Messages"}
                    </ActionBtn>
                    <ActionBtn onClick={() => remove(b)} variant="ghost" disabled={busyId === b.id}>
                      Delete
                    </ActionBtn>
                  </div>
                )}

                {openChatId === b.id && (
                  <div className="mt-4">
                    <MessageThread bookingId={b.id} side="owner" />
                  </div>
                )}
              </li>
            );
          })}
        </ul>
      </section>
      <ChatWidget mode="admin" />
    </main>
  );
}

function ActionBtn({
  children,
  onClick,
  variant = "default",
  disabled,
}: {
  children: React.ReactNode;
  onClick: () => void;
  variant?: "default" | "primary" | "danger" | "ghost";
  disabled?: boolean;
}) {
  const styles =
    variant === "primary"
      ? "bg-deep text-white hover:bg-deep/90"
      : variant === "danger"
      ? "bg-terracotta text-white hover:bg-terracotta/90"
      : variant === "ghost"
      ? "text-deep/70 hover:bg-deep/5"
      : "bg-sand text-deep ring-1 ring-deep/10 hover:bg-deep/5";
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`rounded-full px-4 py-2 text-sm font-medium transition disabled:opacity-50 ${styles}`}
    >
      {children}
    </button>
  );
}

function EditForm({
  booking,
  busy,
  onCancel,
  onSave,
}: {
  booking: AdminBooking;
  busy: boolean;
  onCancel: () => void;
  onSave: (changes: {
    checkIn: string;
    checkOut: string;
    status: AdminBooking["status"];
    notes: string;
    ownerNote: string;
  }) => void;
}) {
  const [checkIn, setCheckIn] = useState(booking.checkIn);
  const [checkOut, setCheckOut] = useState(booking.checkOut);
  const [status, setStatus] = useState<AdminBooking["status"]>(booking.status);
  const [notes, setNotes] = useState(booking.notes ?? "");
  const [ownerNote, setOwnerNote] = useState(booking.ownerNote ?? "");

  return (
    <div className="mt-4 grid gap-3 rounded-xl bg-sand/60 p-4">
      <div className="grid gap-3 sm:grid-cols-3">
        <label className="block">
          <span className="text-xs font-medium text-ink/70">Check-in</span>
          <input
            type="date"
            value={checkIn}
            onChange={(e) => setCheckIn(e.target.value)}
            className="mt-1 w-full rounded-lg border border-ink/15 bg-white px-3 py-2"
          />
        </label>
        <label className="block">
          <span className="text-xs font-medium text-ink/70">Check-out</span>
          <input
            type="date"
            value={checkOut}
            onChange={(e) => setCheckOut(e.target.value)}
            className="mt-1 w-full rounded-lg border border-ink/15 bg-white px-3 py-2"
          />
        </label>
        <label className="block">
          <span className="text-xs font-medium text-ink/70">Status</span>
          <select
            value={status}
            onChange={(e) => setStatus(e.target.value as AdminBooking["status"])}
            className="mt-1 w-full rounded-lg border border-ink/15 bg-white px-3 py-2"
          >
            {STATUS_OPTIONS.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </label>
      </div>
      <label className="block">
        <span className="text-xs font-medium text-ocean">
          Note to guest (shown on their booking when accepted)
        </span>
        <textarea
          value={ownerNote}
          onChange={(e) => setOwnerNote(e.target.value)}
          rows={3}
          placeholder="e.g. Gate code is 4421. Wi-Fi password is on the fridge. We'll leave the keys in the lockbox by the door."
          className="mt-1 w-full rounded-lg border border-sea/30 bg-white px-3 py-2"
        />
      </label>
      <label className="block">
        <span className="text-xs font-medium text-ink/70">Notes (private — guest doesn&apos;t see)</span>
        <textarea
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          rows={2}
          className="mt-1 w-full rounded-lg border border-ink/15 bg-white px-3 py-2"
        />
      </label>
      <div className="flex justify-end gap-2">
        <ActionBtn onClick={onCancel} variant="ghost" disabled={busy}>
          Cancel
        </ActionBtn>
        <ActionBtn
          onClick={() => onSave({ checkIn, checkOut, status, notes, ownerNote })}
          variant="primary"
          disabled={busy}
        >
          {busy ? "Saving…" : "Save & notify guest"}
        </ActionBtn>
      </div>
    </div>
  );
}
