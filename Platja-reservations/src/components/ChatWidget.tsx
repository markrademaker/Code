"use client";

import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { format, parseISO } from "date-fns";
import { MessageThread } from "@/components/MessageThread";

export type ChatMode = "guest" | "admin";

type Conversation = {
  id: string;
  name?: string;
  checkIn: string;
  checkOut: string;
  status: "PENDING" | "TENTATIVE" | "CONFIRMED" | "DECLINED" | "CANCELLED";
  lastMessage: {
    body: string;
    fromOwner: boolean;
    createdAt: string;
  } | null;
};

const STATUS_DOT: Record<Conversation["status"], string> = {
  PENDING: "bg-sunset",
  TENTATIVE: "bg-stone",
  CONFIRMED: "bg-olive",
  DECLINED: "bg-ink/40",
  CANCELLED: "bg-ink/30",
};

export function ChatWidget({ mode }: { mode: ChatMode }) {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [items, setItems] = useState<Conversation[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const apiList =
    mode === "admin" ? "/api/admin/bookings/list" : "/api/my-bookings/list";

  const hide =
    pathname === "/login" ||
    (mode === "guest" && pathname?.startsWith("/admin")) ||
    (mode === "admin" && !pathname?.startsWith("/admin"));

  useEffect(() => {
    if (!open) return;
    let alive = true;
    setError(null);
    fetch(apiList)
      .then(async (r) => {
        if (!r.ok) throw new Error("Could not load");
        return r.json();
      })
      .then((j) => {
        if (!alive) return;
        setItems(j.bookings ?? []);
      })
      .catch((err) => {
        if (!alive) return;
        setError(err instanceof Error ? err.message : "Error");
      });
    return () => {
      alive = false;
    };
  }, [open, apiList]);

  if (hide) return null;

  const active = items?.find((x) => x.id === activeId) ?? null;
  const visibleItems =
    mode === "admin"
      ? items
      : items?.filter(
          (x) => x.status !== "CANCELLED" && x.status !== "DECLINED",
        );

  return (
    <>
      <button
        type="button"
        aria-label={open ? "Close messages" : "Open messages"}
        onClick={() => {
          setOpen((v) => !v);
          setActiveId(null);
        }}
        className="fixed bottom-5 right-5 z-40 flex h-14 w-14 items-center justify-center rounded-full bg-ocean text-2xl text-whitewash shadow-glow transition hover:scale-105 sm:bottom-6 sm:right-6"
      >
        {open ? "×" : "💬"}
      </button>

      {open && (
        <div className="fixed bottom-24 right-4 z-40 flex max-h-[75dvh] w-[calc(100vw-2rem)] max-w-sm flex-col overflow-hidden rounded-3xl bg-white shadow-glow ring-1 ring-ink/10 sm:right-6">
          <div className="flex items-center gap-3 border-b border-ink/5 bg-whitewash px-5 py-4">
            {active ? (
              <button
                aria-label="Back"
                onClick={() => setActiveId(null)}
                className="-ml-1 flex h-8 w-8 items-center justify-center rounded-full hover:bg-ink/5"
              >
                ←
              </button>
            ) : null}
            <div className="min-w-0 flex-1">
              <p className="text-[10px] uppercase tracking-wider text-ink/55">
                {mode === "admin" ? "Owner inbox" : "Messages"}
              </p>
              <p className="truncate font-display text-base font-semibold text-ink">
                {active
                  ? `${mode === "admin" ? active.name + " · " : ""}${format(parseISO(active.checkIn), "d MMM")} → ${format(parseISO(active.checkOut), "d MMM")}`
                  : mode === "admin"
                    ? "All booking threads"
                    : "Your bookings"}
              </p>
            </div>
          </div>

          {active ? (
            <div className="flex-1 overflow-y-auto p-4">
              <MessageThread
                bookingId={active.id}
                side={mode === "admin" ? "owner" : "guest"}
              />
            </div>
          ) : (
            <div className="flex-1 overflow-y-auto">
              {!items && !error && (
                <p className="p-6 text-center text-sm text-ink/55">Loading…</p>
              )}
              {error && (
                <p className="m-4 rounded-2xl bg-terracotta/10 px-4 py-3 text-sm text-terracotta">
                  {error}
                </p>
              )}
              {items && visibleItems?.length === 0 && (
                <p className="p-6 text-center text-sm text-ink/55">
                  {mode === "admin"
                    ? "No bookings yet."
                    : "Request a booking to start a conversation."}
                </p>
              )}
              <ul>
                {visibleItems?.map((c) => (
                  <li key={c.id}>
                    <button
                      onClick={() => setActiveId(c.id)}
                      className="flex w-full gap-3 border-b border-ink/5 px-5 py-4 text-left hover:bg-whitewash"
                    >
                      <span
                        className={`mt-1.5 h-2.5 w-2.5 shrink-0 rounded-full ${STATUS_DOT[c.status]}`}
                        aria-hidden
                      />
                      <span className="min-w-0 flex-1">
                        <span className="flex items-baseline justify-between gap-3">
                          <span className="truncate text-sm font-semibold text-ink">
                            {mode === "admin" ? c.name : `${format(parseISO(c.checkIn), "d MMM")} → ${format(parseISO(c.checkOut), "d MMM")}`}
                          </span>
                          <span className="shrink-0 text-[10px] uppercase tracking-wider text-ink/55">
                            {c.status.toLowerCase()}
                          </span>
                        </span>
                        {mode === "admin" && (
                          <span className="block text-[11px] text-ink/55">
                            {format(parseISO(c.checkIn), "d MMM")} →{" "}
                            {format(parseISO(c.checkOut), "d MMM yyyy")}
                          </span>
                        )}
                        <span className="mt-1 block truncate text-xs text-ink/65">
                          {c.lastMessage
                            ? `${c.lastMessage.fromOwner ? "You: " : ""}${c.lastMessage.body}`
                            : "No messages yet"}
                        </span>
                      </span>
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </>
  );
}
