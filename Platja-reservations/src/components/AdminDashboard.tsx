"use client";

import { useMemo, useState } from "react";
import { format, parseISO } from "date-fns";
import { AdminPlanningCalendar } from "@/components/AdminPlanningCalendar";

export type AdminBooking = {
  id: string;
  name: string;
  email: string;
  phone: string | null;
  guests: number;
  checkIn: string;
  checkOut: string;
  message: string | null;
  status: "PENDING" | "TENTATIVE" | "CONFIRMED" | "DECLINED" | "CANCELLED";
  notes: string | null;
  createdAt: string;
};

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
  const [busyId, setBusyId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

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
    changes: { checkIn: string; checkOut: string; status: AdminBooking["status"]; notes: string },
  ): Promise<void> {
    const ok = await patch(b.id, changes);
    if (ok) {
      refreshAndReplace(b.id, changes);
      setEditingId(null);
    }
  }

  async function remove(b: AdminBooking): Promise<void> {
    if (!confirm(`Delete booking from ${b.name}? This cannot be undone.`)) return;
    setBusyId(b.id);
    const res = await fetch(`/api/admin/bookings/${b.id}`, { method: "DELETE" });
    setBusyId(null);
    if (res.ok) setBookings((bs) => bs.filter((x) => x.id !== b.id));
  }

  async function logout(): Promise<void> {
    await fetch("/api/admin/logout", { method: "POST" });
    window.location.href = "/";
  }

  return (
    <main className="mx-auto max-w-6xl px-5 py-8 sm:px-6 sm:py-12">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className="text-xs uppercase tracking-[0.25em] text-deep/60">Admin</p>
          <h1 className="mt-1 font-display text-3xl font-semibold sm:text-4xl">Planning</h1>
        </div>
        <button
          onClick={logout}
          className="rounded-full bg-white px-4 py-2 text-sm font-medium shadow-sm ring-1 ring-deep/10 hover:bg-deep/5"
        >
          Sign out
        </button>
      </div>

      <AdminPlanningCalendar bookings={bookings} />

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
            return (
              <li
                key={b.id}
                className="rounded-2xl bg-white p-5 shadow-sm ring-1 ring-deep/5"
              >
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <p className="font-semibold text-deep">{b.name}</p>
                    <p className="text-sm text-deep/70">
                      <a href={`mailto:${b.email}`} className="hover:underline">
                        {b.email}
                      </a>
                      {b.phone && <> · {b.phone}</>}
                    </p>
                    <p className="mt-1 text-sm text-deep/80">
                      {format(parseISO(b.checkIn), "EEE d MMM yyyy")}{" → "}
                      {format(parseISO(b.checkOut), "EEE d MMM yyyy")}
                      <span className="ml-2 text-deep/60">· {b.guests} guests</span>
                    </p>
                  </div>
                  <span
                    className={`rounded-full px-3 py-1 text-xs font-medium ${STATUS_STYLES[b.status]}`}
                  >
                    {b.status}
                  </span>
                </div>

                {b.message && (
                  <p className="mt-3 whitespace-pre-wrap rounded-xl bg-sand/60 px-4 py-3 text-sm text-deep/80">
                    {b.message}
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
                      Edit / move
                    </ActionBtn>
                    <ActionBtn onClick={() => remove(b)} variant="ghost" disabled={busyId === b.id}>
                      Delete
                    </ActionBtn>
                  </div>
                )}
              </li>
            );
          })}
        </ul>
      </section>
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
  onSave: (changes: { checkIn: string; checkOut: string; status: AdminBooking["status"]; notes: string }) => void;
}) {
  const [checkIn, setCheckIn] = useState(booking.checkIn);
  const [checkOut, setCheckOut] = useState(booking.checkOut);
  const [status, setStatus] = useState<AdminBooking["status"]>(booking.status);
  const [notes, setNotes] = useState(booking.notes ?? "");

  return (
    <div className="mt-4 grid gap-3 rounded-xl bg-sand/60 p-4">
      <div className="grid gap-3 sm:grid-cols-3">
        <label className="block">
          <span className="text-xs font-medium text-deep/70">Check-in</span>
          <input
            type="date"
            value={checkIn}
            onChange={(e) => setCheckIn(e.target.value)}
            className="mt-1 w-full rounded-lg border border-deep/15 bg-white px-3 py-2"
          />
        </label>
        <label className="block">
          <span className="text-xs font-medium text-deep/70">Check-out</span>
          <input
            type="date"
            value={checkOut}
            onChange={(e) => setCheckOut(e.target.value)}
            className="mt-1 w-full rounded-lg border border-deep/15 bg-white px-3 py-2"
          />
        </label>
        <label className="block">
          <span className="text-xs font-medium text-deep/70">Status</span>
          <select
            value={status}
            onChange={(e) => setStatus(e.target.value as AdminBooking["status"])}
            className="mt-1 w-full rounded-lg border border-deep/15 bg-white px-3 py-2"
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
        <span className="text-xs font-medium text-deep/70">Notes (private)</span>
        <textarea
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          rows={2}
          className="mt-1 w-full rounded-lg border border-deep/15 bg-white px-3 py-2"
        />
      </label>
      <div className="flex justify-end gap-2">
        <ActionBtn onClick={onCancel} variant="ghost" disabled={busy}>
          Cancel
        </ActionBtn>
        <ActionBtn
          onClick={() => onSave({ checkIn, checkOut, status, notes })}
          variant="primary"
          disabled={busy}
        >
          {busy ? "Saving…" : "Save & notify guest"}
        </ActionBtn>
      </div>
    </div>
  );
}
