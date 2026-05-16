"use client";

import { useMemo, useState } from "react";
import {
  addMonths,
  eachDayOfInterval,
  endOfMonth,
  endOfWeek,
  format,
  isSameMonth,
  isWithinInterval,
  parseISO,
  startOfMonth,
  startOfWeek,
} from "date-fns";
import type { AdminBooking } from "@/components/AdminDashboard";

const WEEKDAYS = ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"];

function colorFor(status: AdminBooking["status"]): string {
  switch (status) {
    case "CONFIRMED":
      return "bg-emerald-500/85 text-white";
    case "TENTATIVE":
      return "bg-amber-400/85 text-deep";
    case "PENDING":
      return "bg-yellow-200 text-deep";
    default:
      return "";
  }
}

function bookingForDay(day: Date, bookings: AdminBooking[]): AdminBooking | undefined {
  return bookings.find(
    (b) =>
      (b.status === "CONFIRMED" || b.status === "TENTATIVE" || b.status === "PENDING") &&
      isWithinInterval(day, { start: parseISO(b.checkIn), end: parseISO(b.checkOut) }),
  );
}

function Month({ month, bookings }: { month: Date; bookings: AdminBooking[] }) {
  const days = eachDayOfInterval({
    start: startOfWeek(startOfMonth(month), { weekStartsOn: 1 }),
    end: endOfWeek(endOfMonth(month), { weekStartsOn: 1 }),
  });

  return (
    <div className="rounded-2xl bg-white p-4 shadow-sm ring-1 ring-deep/5">
      <p className="mb-3 text-center font-semibold text-deep">{format(month, "MMMM yyyy")}</p>
      <div className="grid grid-cols-7 gap-1 text-center text-[10px] text-deep/50">
        {WEEKDAYS.map((d) => (
          <div key={d} className="py-1">
            {d}
          </div>
        ))}
      </div>
      <div className="mt-1 grid grid-cols-7 gap-1">
        {days.map((day) => {
          const inMonth = isSameMonth(day, month);
          const booking = inMonth ? bookingForDay(day, bookings) : undefined;
          const baseClasses = "aspect-square flex flex-col items-center justify-center rounded text-[11px] leading-tight";
          const cls = !inMonth
            ? `${baseClasses} text-deep/20`
            : booking
            ? `${baseClasses} ${colorFor(booking.status)}`
            : `${baseClasses} bg-deep/5 text-deep`;
          return (
            <div
              key={day.toISOString()}
              className={cls}
              title={booking ? `${booking.name} (${booking.status.toLowerCase()})` : undefined}
            >
              <span>{format(day, "d")}</span>
              {booking && inMonth && (
                <span className="hidden truncate px-1 sm:block sm:text-[9px]">
                  {booking.name.split(" ")[0]}
                </span>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function AdminPlanningCalendar({ bookings }: { bookings: AdminBooking[] }) {
  const [offset, setOffset] = useState(0);
  const months = useMemo(() => {
    const base = startOfMonth(addMonths(new Date(), offset));
    return [base, addMonths(base, 1), addMonths(base, 2)];
  }, [offset]);

  return (
    <section className="mt-8">
      <div className="flex items-center justify-between">
        <h2 className="font-display text-xl font-semibold sm:text-2xl">Planning</h2>
        <div className="flex gap-2">
          <button
            onClick={() => setOffset((o) => o - 1)}
            className="flex h-10 w-10 items-center justify-center rounded-full bg-white shadow-sm ring-1 ring-deep/10 hover:bg-deep/5"
            aria-label="Previous months"
          >
            ←
          </button>
          <button
            onClick={() => setOffset((o) => o + 1)}
            className="flex h-10 w-10 items-center justify-center rounded-full bg-white shadow-sm ring-1 ring-deep/10 hover:bg-deep/5"
            aria-label="Next months"
          >
            →
          </button>
        </div>
      </div>
      <p className="mt-2 text-sm text-deep/70">
        <span className="mr-3 inline-flex items-center gap-2">
          <span className="inline-block h-3 w-3 rounded-sm bg-emerald-500/85"></span>Confirmed
        </span>
        <span className="mr-3 inline-flex items-center gap-2">
          <span className="inline-block h-3 w-3 rounded-sm bg-amber-400/85"></span>Tentative
        </span>
        <span className="inline-flex items-center gap-2">
          <span className="inline-block h-3 w-3 rounded-sm bg-yellow-200"></span>Pending
        </span>
      </p>
      <div className="mt-4 grid gap-4 md:grid-cols-3">
        {months.map((m) => (
          <Month key={m.toISOString()} month={m} bookings={bookings} />
        ))}
      </div>
    </section>
  );
}
