"use client";

import { useMemo, useState } from "react";
import {
  addMonths,
  eachDayOfInterval,
  endOfMonth,
  format,
  isAfter,
  isSameDay,
  isSameMonth,
  isWithinInterval,
  parseISO,
  startOfMonth,
  startOfWeek,
  endOfWeek,
  isBefore,
  startOfDay,
} from "date-fns";
import type { PublicBooking as Booking } from "@/lib/bookings";
import { SectionMark } from "@/components/Marks";
import { Frost } from "@/components/Frost";

const WEEKDAYS = ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"];

function isBooked(date: Date, bookings: Booking[]): boolean {
  return bookings.some((b) =>
    isWithinInterval(date, { start: parseISO(b.start), end: parseISO(b.end) }),
  );
}

function Month({
  month,
  bookings,
  selectedCheckIn,
  selectedCheckOut,
  onDateClick,
}: {
  month: Date;
  bookings: Booking[];
  selectedCheckIn: Date | null;
  selectedCheckOut: Date | null;
  onDateClick?: (iso: string) => void;
}) {
  const today = startOfDay(new Date());
  const days = eachDayOfInterval({
    start: startOfWeek(startOfMonth(month), { weekStartsOn: 1 }),
    end: endOfWeek(endOfMonth(month), { weekStartsOn: 1 }),
  });

  return (
    <div className="rounded-3xl bg-white p-5 shadow-soft ring-1 ring-ink/5">
      <p className="mb-4 text-center font-display text-lg font-semibold text-ink">
        {format(month, "MMMM yyyy")}
      </p>
      <div className="grid grid-cols-7 gap-1 text-center text-[11px] uppercase tracking-wider text-ink/45">
        {WEEKDAYS.map((d) => (
          <div key={d} className="py-1">
            {d}
          </div>
        ))}
      </div>
      <div className="mt-1 grid grid-cols-7 gap-1.5">
        {days.map((day) => {
          const inMonth = isSameMonth(day, month);
          const past = isBefore(day, today);
          const booked = isBooked(day, bookings);
          const iso = format(day, "yyyy-MM-dd");

          const isStart = !!selectedCheckIn && isSameDay(day, selectedCheckIn);
          const isEnd = !!selectedCheckOut && isSameDay(day, selectedCheckOut);
          const inRange =
            !!selectedCheckIn &&
            !!selectedCheckOut &&
            isAfter(day, selectedCheckIn) &&
            isBefore(day, selectedCheckOut);

          const interactive =
            inMonth && !past && !booked && typeof onDateClick === "function";

          const baseCls =
            "aspect-square flex items-center justify-center rounded-xl text-sm transition";
          const stateCls = !inMonth
            ? "text-ink/15"
            : past
              ? "text-ink/25"
              : booked
                ? "bg-terracotta text-whitewash shadow-soft"
                : isStart || isEnd
                  ? "bg-ocean text-whitewash shadow-soft"
                  : inRange
                    ? "bg-ocean/15 text-ocean"
                    : "bg-sea/10 text-ocean";

          const interactiveCls = interactive
            ? "cursor-pointer hover:ring-2 hover:ring-ocean/40"
            : "";

          if (interactive) {
            return (
              <button
                key={day.toISOString()}
                type="button"
                onClick={() => onDateClick?.(iso)}
                className={`${baseCls} ${stateCls} ${interactiveCls}`}
                aria-label={`Pick ${format(day, "EEEE d MMMM yyyy")}`}
              >
                {format(day, "d")}
              </button>
            );
          }

          return (
            <div
              key={day.toISOString()}
              className={`${baseCls} ${stateCls}`}
              aria-hidden={!inMonth}
            >
              {format(day, "d")}
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function AvailabilityCalendar({
  bookings,
  frostStrength = 80,
  selectedCheckIn,
  selectedCheckOut,
  onDateClick,
}: {
  bookings: Booking[];
  frostStrength?: number;
  selectedCheckIn?: string | null;
  selectedCheckOut?: string | null;
  onDateClick?: (iso: string) => void;
}) {
  const [offset, setOffset] = useState(0);
  const months = useMemo(() => {
    const base = startOfMonth(addMonths(new Date(), offset));
    return [base, addMonths(base, 1), addMonths(base, 2)];
  }, [offset]);

  const ciDate = selectedCheckIn ? parseISO(selectedCheckIn) : null;
  const coDate = selectedCheckOut ? parseISO(selectedCheckOut) : null;

  return (
    <section id="availability" className="relative mx-auto max-w-7xl px-5 sm:px-8">
      <Frost strength={frostStrength} className="p-8 sm:p-12 lg:p-16">
        <div className="flex flex-wrap items-end justify-between gap-6">
          <div className="max-w-xl">
            <SectionMark number="II" label="When are you free?" />
            <h2 className="mt-4 font-display text-4xl font-light leading-[1.05] tracking-tightish sm:text-5xl lg:text-6xl">
              Pick a <span>week</span>.
            </h2>
            {onDateClick && (
              <p className="mt-3 text-sm text-ink/60">
                {!ciDate
                  ? "Tap an available date to set your check-in."
                  : !coDate
                    ? "Now tap your check-out date."
                    : `${format(ciDate, "d MMM")} → ${format(coDate, "d MMM")}.`}
              </p>
            )}
          </div>
          <div className="flex gap-2">
            <button
              onClick={() => setOffset((o) => o - 1)}
              className="flex h-11 w-11 items-center justify-center rounded-full bg-white text-base shadow-soft ring-1 ring-ink/10 hover:bg-whitewash"
              aria-label="Previous months"
            >
              ←
            </button>
            <button
              onClick={() => setOffset((o) => o + 1)}
              className="flex h-11 w-11 items-center justify-center rounded-full bg-white text-base shadow-soft ring-1 ring-ink/10 hover:bg-whitewash"
              aria-label="Next months"
            >
              →
            </button>
          </div>
        </div>
        <p className="mt-4 text-sm text-ink/65">
          <span className="mr-4 inline-flex items-center gap-2">
            <span className="inline-block h-3 w-3 rounded-md bg-sea/30"></span>
            Available
          </span>
          <span className="mr-4 inline-flex items-center gap-2">
            <span className="inline-block h-3 w-3 rounded-md bg-terracotta"></span>
            Booked
          </span>
          <span className="inline-flex items-center gap-2">
            <span className="inline-block h-3 w-3 rounded-md bg-ocean"></span>
            Your dates
          </span>
        </p>
        <div className="mt-6 grid gap-4 sm:mt-8 sm:gap-6 md:grid-cols-3">
          {months.map((m) => (
            <Month
              key={m.toISOString()}
              month={m}
              bookings={bookings}
              selectedCheckIn={ciDate}
              selectedCheckOut={coDate}
              onDateClick={onDateClick}
            />
          ))}
        </div>
      </Frost>
    </section>
  );
}
