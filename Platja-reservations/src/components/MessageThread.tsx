"use client";

import { useEffect, useState } from "react";
import { format, parseISO } from "date-fns";

type Message = {
  id: string;
  body: string;
  fromOwner: boolean;
  authorName: string;
  createdAt: string;
};

export function MessageThread({
  bookingId,
  side,
}: {
  bookingId: string;
  side: "guest" | "owner";
}) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const apiBase =
    side === "guest"
      ? `/api/my-bookings/${bookingId}/messages`
      : `/api/admin/bookings/${bookingId}/messages`;

  useEffect(() => {
    let alive = true;
    fetch(apiBase)
      .then((r) => r.json())
      .then((j) => {
        if (!alive) return;
        setMessages(j.messages ?? []);
        setLoaded(true);
      })
      .catch(() => setLoaded(true));
    return () => {
      alive = false;
    };
  }, [apiBase]);

  async function send(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const form = e.currentTarget;
    const data = new FormData(form);
    const body = String(data.get("body") ?? "").trim();
    if (!body) return;
    setSending(true);
    setError(null);
    const res = await fetch(apiBase, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ body }),
    });
    setSending(false);
    if (!res.ok) {
      setError("Could not send");
      return;
    }
    const j = await res.json();
    setMessages((m) => [...m, j.message]);
    form.reset();
  }

  return (
    <div className="rounded-2xl bg-whitewash p-4">
      <p className="text-xs uppercase tracking-wider text-ink/55">Messages</p>
      <div className="mt-3 grid gap-2">
        {!loaded && <p className="text-sm text-ink/55">Loading…</p>}
        {loaded && messages.length === 0 && (
          <p className="text-sm text-ink/55">
            No messages yet. Start the conversation below.
          </p>
        )}
        {messages.map((m) => {
          const mine =
            (side === "guest" && !m.fromOwner) ||
            (side === "owner" && m.fromOwner);
          return (
            <div
              key={m.id}
              className={`flex ${mine ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`max-w-[85%] rounded-2xl px-3 py-2 text-sm ${
                  mine
                    ? "bg-ocean text-whitewash"
                    : m.fromOwner
                      ? "bg-sand text-ink"
                      : "bg-white text-ink ring-1 ring-ink/10"
                }`}
              >
                <p className="whitespace-pre-wrap leading-snug">{m.body}</p>
                <p
                  className={`mt-1 text-[10px] ${
                    mine ? "text-whitewash/70" : "text-ink/50"
                  }`}
                >
                  {m.authorName} ·{" "}
                  {format(parseISO(m.createdAt), "d MMM HH:mm")}
                </p>
              </div>
            </div>
          );
        })}
      </div>
      <form onSubmit={send} className="mt-3 flex gap-2">
        <input
          name="body"
          placeholder={
            side === "guest" ? "Message the owners…" : "Reply to guest…"
          }
          className="flex-1 rounded-full border border-ink/15 bg-white px-4 py-2 text-sm focus:border-sea focus:outline-none focus:ring-2 focus:ring-sea/30"
          maxLength={2000}
        />
        <button
          type="submit"
          disabled={sending}
          className="rounded-full bg-ocean px-5 text-sm font-medium text-whitewash disabled:opacity-50"
        >
          {sending ? "…" : "Send"}
        </button>
      </form>
      {error && (
        <p className="mt-2 rounded-xl bg-terracotta/10 px-3 py-2 text-xs text-terracotta">
          {error}
        </p>
      )}
    </div>
  );
}
