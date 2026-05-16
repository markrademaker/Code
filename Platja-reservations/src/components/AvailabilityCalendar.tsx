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
    <div className="rounded-2xl bg-white p-4 shadow-sm ring-1 ring-deep/5 sm:p-5">
      <p className="mb-3 text-center font-semibold text-deep sm:mb-4">
        {format(month, "MMMM yyyy")}
      </p>
      <div className="grid grid-cols-7 gap-1 text-center text-xs text-deep/50">
        {WEEKDAYS.map((d) => (
          <div key={d} className="py-1">
            {d}
          </div>
        ))}
      </div>
      <div className="mt-1 grid grid-cols-7 gap-1">
        {days.map((day) => {
          const inMonth = isSameMonth(day, month);
          const past = isBefore(day, today);
          const booked = isBooked(day, bookings);
          const classes = [
            "aspect-square flex items-center justify-center rounded-md text-sm",
            !inMonth && "text-deep/20",
            inMonth && past && "text-deep/30",
            inMonth && !past && booked && "bg-terracotta/90 text-white",
            inMonth && !past && !booked && "bg-sea/10 text-deep",
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
    <section id="availability" className="bg-sand py-14 sm:py-20">
      <div className="mx-auto max-w-6xl px-5 sm:px-6">
        <div className="flex items-center justify-between gap-3">
          <h2 className="font-display text-2xl font-semibold sm:text-3xl lg:text-4xl">
            Availability
          </h2>
          <div className="flex gap-2">
            <button
              onClick={() => setOffset((o) => o - 1)}
              className="flex h-11 w-11 items-center justify-center rounded-full bg-white text-base shadow-sm ring-1 ring-deep/10 hover:bg-deep/5"
              aria-label="Previous months"
            >
              ←
            </button>
            <button
              onClick={() => setOffset((o) => o + 1)}
              className="flex h-11 w-11 items-center justify-center rounded-full bg-white text-base shadow-sm ring-1 ring-deep/10 hover:bg-deep/5"
              aria-label="Next months"
            >
              →
            </button>
          </div>
        </div>
        <p className="mt-3 text-sm text-deep/70 sm:text-base">
          <span className="mr-3 inline-flex items-center gap-2">
            <span className="inline-block h-3 w-3 rounded-sm bg-sea/30"></span>
            Available
          </span>
          <span className="inline-flex items-center gap-2">
            <span className="inline-block h-3 w-3 rounded-sm bg-terracotta/90"></span>
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
