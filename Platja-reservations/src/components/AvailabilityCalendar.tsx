"use client";

import { useMemo, useState } from "react";
import {
  addMonths,
  eachDayOfInterval,
  endOfMonth,
  format,
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

const WEEKDAYS = ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"];

function isBooked(date: Date, bookings: Booking[]): boolean {
  return bookings.some((b) =>
    isWithinInterval(date, { start: parseISO(b.start), end: parseISO(b.end) }),
  );
}

function Month({
  month,
  bookings,
}: {
  month: Date;
  bookings: Booking[];
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
          const classes = [
            "aspect-square flex items-center justify-center rounded-xl text-sm transition",
            !inMonth && "text-ink/15",
            inMonth && past && "text-ink/25",
            inMonth && !past && booked && "bg-terracotta text-whitewash shadow-soft",
            inMonth && !past && !booked && "bg-sea/10 text-ocean",
          ]
            .filter(Boolean)
            .join(" ");
          return (
            <div key={day.toISOString()} className={classes}>
              {format(day, "d")}
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function AvailabilityCalendar({ bookings }: { bookings: Booking[] }) {
  const [offset, setOffset] = useState(0);
  const months = useMemo(() => {
    const base = startOfMonth(addMonths(new Date(), offset));
    return [base, addMonths(base, 1), addMonths(base, 2)];
  }, [offset]);

  return (
    <section id="availability" className="bg-sand/40 py-20 sm:py-28">
      <div className="mx-auto max-w-7xl px-5 sm:px-8">
        <div className="flex flex-wrap items-end justify-between gap-6">
          <div className="max-w-xl">
            <SectionMark number="II" label="When are you free?" />
            <h2 className="mt-4 font-display text-4xl font-semibold leading-tight sm:text-6xl">
              Pick a <span className="italic text-terracotta">week</span>.
            </h2>
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
          <span className="inline-flex items-center gap-2">
            <span className="inline-block h-3 w-3 rounded-md bg-terracotta"></span>
            Booked
          </span>
        </p>
        <div className="mt-6 grid gap-4 sm:mt-8 sm:gap-6 md:grid-cols-3">
          {months.map((m) => (
            <Month key={m.toISOString()} month={m} bookings={bookings} />
          ))}
        </div>
      </div>
    </section>
  );
}
