"use client";

import Link from "next/link";
import { useState } from "react";
import { ChatWidget } from "@/components/ChatWidget";

export type RestaurantRow = {
  id: string;
  name: string;
  website: string | null;
  phone: string | null;
  area: string | null;
  blurb: string | null;
  sortOrder: number;
};

type Draft = {
  name: string;
  website: string;
  phone: string;
  area: string;
  blurb: string;
  sortOrder: string;
};

const EMPTY: Draft = {
  name: "",
  website: "",
  phone: "",
  area: "",
  blurb: "",
  sortOrder: "0",
};

export function RestaurantAdmin({ initial }: { initial: RestaurantRow[] }) {
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
    const res = await fetch("/api/admin/restaurants", {
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
    setRows((rs) => [
      ...rs,
      {
        id,
        name: draft.name,
        website: draft.website || null,
        phone: draft.phone || null,
        area: draft.area || null,
        blurb: draft.blurb || null,
        sortOrder: Number(draft.sortOrder) || 0,
      },
    ]);
    setDraft(EMPTY);
  }

  function startEdit(r: RestaurantRow) {
    setEditingId(r.id);
    setEdit({
      name: r.name,
      website: r.website ?? "",
      phone: r.phone ?? "",
      area: r.area ?? "",
      blurb: r.blurb ?? "",
      sortOrder: String(r.sortOrder),
    });
  }

  async function saveEdit(id: string) {
    setBusy(true);
    setError(null);
    const res = await fetch(`/api/admin/restaurants/${id}`, {
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
      rs.map((r) =>
        r.id === id
          ? {
              ...r,
              name: edit.name,
              website: edit.website || null,
              phone: edit.phone || null,
              area: edit.area || null,
              blurb: edit.blurb || null,
              sortOrder: Number(edit.sortOrder) || 0,
            }
          : r,
      ),
    );
    setEditingId(null);
  }

  async function remove(id: string) {
    if (!confirm("Delete this restaurant?")) return;
    const res = await fetch(`/api/admin/restaurants/${id}`, { method: "DELETE" });
    if (res.ok) setRows((rs) => rs.filter((r) => r.id !== id));
  }

  return (
    <main className="mx-auto max-w-4xl px-5 py-10 sm:px-6 sm:py-12">
      <p className="text-xs uppercase tracking-[0.2em] text-ink/55">Admin</p>
      <h1 className="mt-1 font-display text-3xl font-semibold sm:text-4xl">
        Restaurant book
      </h1>
      <Link href="/admin" className="mt-2 inline-block text-sm text-ocean hover:underline">
        ← Back to bookings
      </Link>

      <section className="mt-8 rounded-3xl bg-white p-6 shadow-soft ring-1 ring-ink/5 sm:p-7">
        <h2 className="font-display text-xl font-semibold">Add a restaurant</h2>
        <form onSubmit={create} className="mt-4 grid gap-3 sm:grid-cols-2">
          <Field label="Name" value={draft.name} onChange={(v) => setDraft({ ...draft, name: v })} required />
          <Field label="Area (optional)" value={draft.area} onChange={(v) => setDraft({ ...draft, area: v })} />
          <Field label="Website (optional)" value={draft.website} onChange={(v) => setDraft({ ...draft, website: v })} placeholder="https://" />
          <Field label="Phone (optional)" value={draft.phone} onChange={(v) => setDraft({ ...draft, phone: v })} placeholder="+34 …" />
          <label className="sm:col-span-2 block">
            <span className="text-sm font-medium text-ink">Blurb (optional)</span>
            <textarea
              rows={2}
              value={draft.blurb}
              onChange={(e) => setDraft({ ...draft, blurb: e.target.value })}
              className="mt-1 w-full rounded-xl border border-ink/15 bg-white px-3 py-2"
            />
          </label>
          {error && <p className="sm:col-span-2 rounded-xl bg-terracotta/10 px-4 py-3 text-terracotta">{error}</p>}
          <div className="sm:col-span-2 flex justify-end">
            <button
              type="submit"
              disabled={busy || !draft.name}
              className="rounded-full bg-ocean px-6 py-2.5 font-medium text-whitewash disabled:opacity-50"
            >
              {busy ? "Saving…" : "Add"}
            </button>
          </div>
        </form>
      </section>

      <section className="mt-8">
        <h2 className="font-display text-xl font-semibold">Current list ({rows.length})</h2>
        <ul className="mt-4 grid gap-3">
          {rows.map((r) => {
            const editing = editingId === r.id;
            return (
              <li key={r.id} className="rounded-3xl bg-white p-5 shadow-soft ring-1 ring-ink/5">
                {editing ? (
                  <div className="grid gap-3 sm:grid-cols-2">
                    <Field label="Name" value={edit.name} onChange={(v) => setEdit({ ...edit, name: v })} />
                    <Field label="Area" value={edit.area} onChange={(v) => setEdit({ ...edit, area: v })} />
                    <Field label="Website" value={edit.website} onChange={(v) => setEdit({ ...edit, website: v })} />
                    <Field label="Phone" value={edit.phone} onChange={(v) => setEdit({ ...edit, phone: v })} />
                    <label className="sm:col-span-2 block">
                      <span className="text-sm font-medium text-ink">Blurb</span>
                      <textarea
                        rows={2}
                        value={edit.blurb}
                        onChange={(e) => setEdit({ ...edit, blurb: e.target.value })}
                        className="mt-1 w-full rounded-xl border border-ink/15 bg-white px-3 py-2"
                      />
                    </label>
                    <div className="sm:col-span-2 flex justify-end gap-2">
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
                  <div className="flex flex-wrap items-start justify-between gap-3">
                    <div className="min-w-0 flex-1">
                      {r.area && <p className="text-xs uppercase tracking-wider text-ink/55">{r.area}</p>}
                      <p className="mt-0.5 font-display text-lg font-semibold text-ink">{r.name}</p>
                      {r.blurb && <p className="mt-1 text-sm text-ink/70">{r.blurb}</p>}
                      <div className="mt-2 flex flex-wrap gap-2 text-xs">
                        {r.website && <span className="rounded-full bg-sea/10 px-2.5 py-0.5 text-ocean">{r.website}</span>}
                        {r.phone && <span className="rounded-full bg-sand/60 px-2.5 py-0.5 text-ink">{r.phone}</span>}
                      </div>
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
