"use client";

import { useMemo, useState } from "react";
import {
  addMonths,
  eachDayOfInterval,
  endOfMonth,
  endOfWeek,
  format,
  isAfter,
  isBefore,
  isSameDay,
  isSameMonth,
  parseISO,
  startOfMonth,
  startOfWeek,
} from "date-fns";
import { formatEuro } from "@/lib/pricing";

const WEEKDAYS = ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"];

export type RateRow = {
  id: string;
  startDate: string;
  endDate: string;
  nightlyRateCents: number;
  label: string | null;
};

function findRate(day: Date, rates: RateRow[]): RateRow | undefined {
  return rates.find((r) => {
    const s = parseISO(r.startDate);
    const e = parseISO(r.endDate);
    return !isBefore(day, s) && !isAfter(day, e);
  });
}

function Month({
  month,
  rates,
  selectedStart,
  selectedEnd,
  onClick,
}: {
  month: Date;
  rates: RateRow[];
  selectedStart: Date | null;
  selectedEnd: Date | null;
  onClick: (iso: string) => void;
}) {
  const days = eachDayOfInterval({
    start: startOfWeek(startOfMonth(month), { weekStartsOn: 1 }),
    end: endOfWeek(endOfMonth(month), { weekStartsOn: 1 }),
  });

  return (
    <div className="rounded-2xl bg-white p-4 shadow-soft ring-1 ring-ink/5">
      <p className="mb-3 text-center font-semibold text-ink">
        {format(month, "MMMM yyyy")}
      </p>
      <div className="grid grid-cols-7 gap-1 text-center text-[10px] text-ink/50">
        {WEEKDAYS.map((d) => (
          <div key={d} className="py-1">
            {d}
          </div>
        ))}
      </div>
      <div className="mt-1 grid grid-cols-7 gap-1">
        {days.map((day) => {
          const inMonth = isSameMonth(day, month);
          const rate = inMonth ? findRate(day, rates) : undefined;
          const iso = format(day, "yyyy-MM-dd");

          const isStart = selectedStart && isSameDay(day, selectedStart);
          const isEnd = selectedEnd && isSameDay(day, selectedEnd);
          const inRange =
            selectedStart &&
            selectedEnd &&
            isAfter(day, selectedStart) &&
            isBefore(day, selectedEnd);

          const base =
            "aspect-square flex flex-col items-center justify-center rounded text-[11px] leading-tight";
          const cls = !inMonth
            ? `${base} text-ink/20`
            : isStart || isEnd
              ? `${base} bg-ocean text-whitewash cursor-pointer transition hover:scale-[1.05] hover:ring-2 hover:ring-ocean/30`
              : inRange
                ? `${base} bg-ocean/20 text-ocean cursor-pointer transition hover:scale-[1.05] hover:ring-2 hover:ring-ocean/30`
                : rate
                  ? `${base} bg-sea/15 text-ocean cursor-pointer transition hover:scale-[1.05] hover:ring-2 hover:ring-ink/20`
                  : `${base} bg-ink/5 text-ink cursor-pointer transition hover:scale-[1.05] hover:ring-2 hover:ring-ink/20`;

          if (!inMonth) {
            return (
              <div key={day.toISOString()} className={cls}>
                <span>{format(day, "d")}</span>
              </div>
            );
          }

          return (
            <button
              key={day.toISOString()}
              type="button"
              onClick={() => onClick(iso)}
              className={cls}
              title={rate ? `${formatEuro(rate.nightlyRateCents)} / night` : "No rate"}
            >
              <span>{format(day, "d")}</span>
              {rate && (
                <span className="hidden truncate px-1 sm:block sm:text-[9px]">
                  {formatEuro(rate.nightlyRateCents)}
                </span>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}

export function RateCalendar({
  rates,
  onSavePeriod,
}: {
  rates: RateRow[];
  onSavePeriod: (
    start: string,
    end: string,
    euro: number,
    label?: string,
  ) => Promise<string | null>;
}) {
  const [offset, setOffset] = useState(0);
  const [start, setStart] = useState<string | null>(null);
  const [end, setEnd] = useState<string | null>(null);
  const [euro, setEuro] = useState("");
  const [label, setLabel] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const months = useMemo(() => {
    const base = startOfMonth(addMonths(new Date(), offset));
    return [base, addMonths(base, 1), addMonths(base, 2)];
  }, [offset]);

  function handleClick(iso: string) {
    setError(null);
    if (!start || (start && end)) {
      setStart(iso);
      setEnd(null);
      return;
    }
    if (iso < start) {
      setStart(iso);
      setEnd(null);
      return;
    }
    setEnd(iso);
  }

  async function save() {
    if (!start || !end) {
      setError("Pick a start and end date on the calendar first.");
      return;
    }
    const amount = Number(euro);
    if (!Number.isFinite(amount) || amount <= 0) {
      setError("Enter a nightly rate in euros.");
      return;
    }
    setSaving(true);
    const err = await onSavePeriod(start, end, amount, label.trim() || undefined);
    setSaving(false);
    if (err === null) {
      setStart(null);
      setEnd(null);
      setEuro("");
      setLabel("");
    } else {
      setError(err);
    }
  }

  function reset() {
    setStart(null);
    setEnd(null);
    setEuro("");
    setLabel("");
    setError(null);
  }

  const startDate = start ? parseISO(start) : null;
  const endDate = end ? parseISO(end) : null;

  return (
    <section className="rounded-3xl bg-white p-6 shadow-soft ring-1 ring-ink/5 sm:p-7">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 className="font-display text-xl font-semibold">Rate calendar</h2>
          <p className="mt-1 text-sm text-ink/65">
            {!start
              ? "Tap the start of a period."
              : !end
                ? "Now tap the end of the period."
                : `${format(parseISO(start), "d MMM")} → ${format(parseISO(end), "d MMM yyyy")}`}
          </p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => setOffset((o) => o - 1)}
            className="flex h-10 w-10 items-center justify-center rounded-full bg-whitewash text-base ring-1 ring-ink/10 hover:bg-ink/5"
            aria-label="Previous months"
          >
            ←
          </button>
          <button
            onClick={() => setOffset((o) => o + 1)}
            className="flex h-10 w-10 items-center justify-center rounded-full bg-whitewash text-base ring-1 ring-ink/10 hover:bg-ink/5"
            aria-label="Next months"
          >
            →
          </button>
        </div>
      </div>

      <p className="mt-3 text-xs text-ink/55">
        <span className="mr-3 inline-flex items-center gap-2">
          <span className="inline-block h-3 w-3 rounded bg-ink/5" />
          No rate set
        </span>
        <span className="mr-3 inline-flex items-center gap-2">
          <span className="inline-block h-3 w-3 rounded bg-sea/15" />
          Priced
        </span>
        <span className="inline-flex items-center gap-2">
          <span className="inline-block h-3 w-3 rounded bg-ocean" />
          Your selection
        </span>
      </p>

      <div className="mt-5 grid gap-4 md:grid-cols-3">
        {months.map((m) => (
          <Month
            key={m.toISOString()}
            month={m}
            rates={rates}
            selectedStart={startDate}
            selectedEnd={endDate}
            onClick={handleClick}
          />
        ))}
      </div>

      <div className="mt-6 grid gap-3 rounded-2xl bg-whitewash p-4 sm:grid-cols-[1fr_1fr_auto] sm:items-end sm:p-5">
        <label className="block">
          <span className="text-xs font-medium uppercase tracking-wider text-ink/55">
            € / night
          </span>
          <input
            type="number"
            min={0}
            step={1}
            value={euro}
            onChange={(e) => setEuro(e.target.value)}
            placeholder="350"
            className="mt-1 w-full rounded-xl border border-ink/15 bg-white px-3 py-2"
          />
        </label>
        <label className="block">
          <span className="text-xs font-medium uppercase tracking-wider text-ink/55">
            Label (optional)
          </span>
          <input
            value={label}
            onChange={(e) => setLabel(e.target.value)}
            placeholder="High season"
            className="mt-1 w-full rounded-xl border border-ink/15 bg-white px-3 py-2"
          />
        </label>
        <div className="flex gap-2 sm:justify-end">
          {(start || end || euro || label) && (
            <button
              type="button"
              onClick={reset}
              className="rounded-full px-4 py-2.5 text-sm text-ink/60 hover:bg-ink/5"
            >
              Clear
            </button>
          )}
          <button
            type="button"
            onClick={save}
            disabled={saving || !start || !end || !euro}
            className="rounded-full bg-ocean px-6 py-2.5 text-sm font-medium text-whitewash disabled:opacity-50"
          >
            {saving ? "Saving…" : "Save period"}
          </button>
        </div>
        {error && (
          <p className="rounded-xl bg-terracotta/10 px-4 py-2 text-sm text-terracotta sm:col-span-3">
            {error}
          </p>
        )}
      </div>
    </section>
  );
}
