"use client";

import Link from "next/link";
import { useState } from "react";
import { format, parseISO } from "date-fns";
import { ChatWidget } from "@/components/ChatWidget";
import { RateCalendar } from "@/components/RateCalendar";
import { formatEuro } from "@/lib/pricing";

export type RateRow = {
  id: string;
  startDate: string;
  endDate: string;
  nightlyRateCents: number;
  label: string | null;
};

type Draft = {
  startDate: string;
  endDate: string;
  nightlyEuro: string;
  label: string;
};

const EMPTY: Draft = { startDate: "", endDate: "", nightlyEuro: "", label: "" };

export function RateAdmin({ initial }: { initial: RateRow[] }) {
  const [rows, setRows] = useState(initial);
  const [draft, setDraft] = useState<Draft>(EMPTY);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [edit, setEdit] = useState<Draft>(EMPTY);

  async function create(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    const res = await fetch("/api/admin/rates", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(draft),
    });
    setBusy(false);
    if (!res.ok) {
      const j = await res.json().catch(() => ({}));
      setError(j.error ?? "Could not add");
      return;
    }
    const { id } = await res.json();
    setRows((rs) =>
      [
        ...rs,
        {
          id,
          startDate: draft.startDate,
          endDate: draft.endDate,
          nightlyRateCents: Math.round(Number(draft.nightlyEuro) * 100),
          label: draft.label || null,
        },
      ].sort((a, b) => a.startDate.localeCompare(b.startDate)),
    );
    setDraft(EMPTY);
  }

  async function saveFromCalendar(
    startDate: string,
    endDate: string,
    nightlyEuro: number,
    label?: string,
  ): Promise<boolean> {
    const res = await fetch("/api/admin/rates", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ startDate, endDate, nightlyEuro, label }),
    });
    if (!res.ok) return false;
    const { id } = await res.json();
    setRows((rs) =>
      [
        ...rs,
        {
          id,
          startDate,
          endDate,
          nightlyRateCents: Math.round(nightlyEuro * 100),
          label: label ?? null,
        },
      ].sort((a, b) => a.startDate.localeCompare(b.startDate)),
    );
    return true;
  }

  function startEdit(r: RateRow) {
    setEditingId(r.id);
    setEdit({
      startDate: r.startDate,
      endDate: r.endDate,
      nightlyEuro: String(r.nightlyRateCents / 100),
      label: r.label ?? "",
    });
  }

  async function saveEdit(id: string) {
    setBusy(true);
    setError(null);
    const res = await fetch(`/api/admin/rates/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(edit),
    });
    setBusy(false);
    if (!res.ok) {
      setError("Could not save");
      return;
    }
    setRows((rs) =>
      rs
        .map((r) =>
          r.id === id
            ? {
                ...r,
                startDate: edit.startDate,
                endDate: edit.endDate,
                nightlyRateCents: Math.round(Number(edit.nightlyEuro) * 100),
                label: edit.label || null,
              }
            : r,
        )
        .sort((a, b) => a.startDate.localeCompare(b.startDate)),
    );
    setEditingId(null);
  }

  async function remove(id: string) {
    if (!confirm("Delete this rate period?")) return;
    const res = await fetch(`/api/admin/rates/${id}`, { method: "DELETE" });
    if (res.ok) setRows((rs) => rs.filter((r) => r.id !== id));
  }

  return (
    <main className="mx-auto max-w-4xl px-5 py-10 sm:px-6 sm:py-12">
      <p className="text-xs uppercase tracking-[0.2em] text-ink/55">Admin</p>
      <h1 className="mt-1 font-display text-3xl font-semibold sm:text-4xl">
        Nightly rates
      </h1>
      <Link
        href="/admin"
        className="mt-2 inline-block text-sm text-ocean hover:underline"
      >
        ← Back to bookings
      </Link>

      <div className="mt-8">
        <RateCalendar rates={rows} onSavePeriod={saveFromCalendar} />
      </div>

      <section className="mt-8 rounded-3xl bg-white p-6 shadow-soft ring-1 ring-ink/5 sm:p-7">
        <h2 className="font-display text-xl font-semibold">Add a period (form)</h2>
        <p className="mt-1 text-sm text-ink/65">
          Same as the calendar above — typing in dates and a rate also works.
        </p>
        <form onSubmit={create} className="mt-4 grid gap-3 sm:grid-cols-4">
          <Date label="Start" value={draft.startDate} onChange={(v) => setDraft({ ...draft, startDate: v })} required />
          <Date label="End" value={draft.endDate} onChange={(v) => setDraft({ ...draft, endDate: v })} required />
          <Field label="€ / night" value={draft.nightlyEuro} onChange={(v) => setDraft({ ...draft, nightlyEuro: v })} required placeholder="350" />
          <Field label="Label" value={draft.label} onChange={(v) => setDraft({ ...draft, label: v })} placeholder="Mid-season" />
          {error && (
            <p className="rounded-xl bg-terracotta/10 px-4 py-3 text-terracotta sm:col-span-4">
              {error}
            </p>
          )}
          <div className="sm:col-span-4 flex justify-end">
            <button
              type="submit"
              disabled={busy || !draft.startDate || !draft.endDate || !draft.nightlyEuro}
              className="rounded-full bg-ocean px-6 py-2.5 font-medium text-whitewash disabled:opacity-50"
            >
              {busy ? "Saving…" : "Add"}
            </button>
          </div>
        </form>
      </section>

      <section className="mt-8">
        <h2 className="font-display text-xl font-semibold">
          Configured periods ({rows.length})
        </h2>
        <ul className="mt-4 grid gap-3">
          {rows.length === 0 && (
            <li className="rounded-3xl bg-white p-6 text-center text-ink/60 shadow-soft ring-1 ring-ink/5">
              No rates yet. Nights without a rate will be marked &quot;price on
              request&quot; in bookings.
            </li>
          )}
          {rows.map((r) => {
            const editing = editingId === r.id;
            return (
              <li
                key={r.id}
                className="rounded-3xl bg-white p-5 shadow-soft ring-1 ring-ink/5"
              >
                {editing ? (
                  <div className="grid gap-3 sm:grid-cols-4">
                    <Date label="Start" value={edit.startDate} onChange={(v) => setEdit({ ...edit, startDate: v })} />
                    <Date label="End" value={edit.endDate} onChange={(v) => setEdit({ ...edit, endDate: v })} />
                    <Field label="€ / night" value={edit.nightlyEuro} onChange={(v) => setEdit({ ...edit, nightlyEuro: v })} />
                    <Field label="Label" value={edit.label} onChange={(v) => setEdit({ ...edit, label: v })} />
                    <div className="sm:col-span-4 flex justify-end gap-2">
                      <button onClick={() => setEditingId(null)} className="rounded-full px-4 py-2 text-sm text-ink/60 hover:bg-ink/5">
                        Cancel
                      </button>
                      <button
                        onClick={() => saveEdit(r.id)}
                        disabled={busy}
                        className="rounded-full bg-ocean px-5 py-2 text-sm font-medium text-whitewash disabled:opacity-50"
                      >
                        Save
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div>
                      <p className="font-display text-lg font-semibold text-ink">
                        {format(parseISO(r.startDate), "d MMM yyyy")} →{" "}
                        {format(parseISO(r.endDate), "d MMM yyyy")}
                      </p>
                      <p className="mt-0.5 text-sm text-ink/70">
                        {formatEuro(r.nightlyRateCents)} / night
                        {r.label && (
                          <span className="ml-2 rounded-full bg-sand/60 px-2.5 py-0.5 text-[11px] uppercase tracking-wider text-ink/70">
                            {r.label}
                          </span>
                        )}
                      </p>
                    </div>
                    <div className="flex gap-2">
                      <button onClick={() => startEdit(r)} className="rounded-full bg-sand px-4 py-2 text-sm font-medium text-ink ring-1 ring-ink/10 hover:bg-ink/5">
                        Edit
                      </button>
                      <button onClick={() => remove(r.id)} className="rounded-full px-4 py-2 text-sm text-terracotta hover:bg-terracotta/10">
                        Delete
                      </button>
                    </div>
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

function Field({
  label,
  value,
  onChange,
  required,
  placeholder,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  required?: boolean;
  placeholder?: string;
}) {
  return (
    <label className="block">
      <span className="text-sm font-medium text-ink">{label}</span>
      <input
        value={value}
        onChange={(e) => onChange(e.target.value)}
        required={required}
        placeholder={placeholder}
        className="mt-1 w-full rounded-xl border border-ink/15 bg-white px-3 py-2"
      />
    </label>
  );
}

function Date({
  label,
  value,
  onChange,
  required,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  required?: boolean;
}) {
  return (
    <label className="block">
      <span className="text-sm font-medium text-ink">{label}</span>
      <input
        type="date"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        required={required}
        className="mt-1 w-full rounded-xl border border-ink/15 bg-white px-3 py-2"
      />
    </label>
  );
}
