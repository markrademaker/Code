"use client";

import { useState } from "react";
import { addDays, format, isBefore, parseISO, startOfDay } from "date-fns";
import { AvailabilityCalendar } from "@/components/AvailabilityCalendar";
import { BookingForm, type BookingFormUser } from "@/components/BookingForm";
import type { PublicBooking } from "@/lib/bookings";

function rangeOverlapsBooking(
  start: string,
  end: string,
  bookings: PublicBooking[],
): boolean {
  const s = parseISO(start);
  const e = parseISO(end);
  return bookings.some((b) => {
    const bs = parseISO(b.start);
    const be = parseISO(b.end);
    return !(isBefore(e, bs) || isBefore(be, s));
  });
}

function isBookedDay(day: string, bookings: PublicBooking[]): boolean {
  const d = parseISO(day);
  return bookings.some((b) => {
    const bs = parseISO(b.start);
    const be = parseISO(b.end);
    return !isBefore(d, bs) && !isBefore(be, d);
  });
}

export function BookingFlow({
  bookings,
  user,
  frostStrength = 80,
}: {
  bookings: PublicBooking[];
  user: BookingFormUser | null;
  frostStrength?: number;
}) {
  const [checkIn, setCheckIn] = useState<string>("");
  const [checkOut, setCheckOut] = useState<string>("");

  function handleDateClick(dateIso: string) {
    const today = startOfDay(new Date());
    const date = parseISO(dateIso);
    if (isBefore(date, today)) return;
    if (isBookedDay(dateIso, bookings)) return;

    if (!checkIn || checkOut) {
      setCheckIn(dateIso);
      setCheckOut("");
      return;
    }

    if (!isBefore(parseISO(checkIn), date)) {
      setCheckIn(dateIso);
      setCheckOut("");
      return;
    }

    const tentativeStart = checkIn;
    const tentativeEnd = format(addDays(date, -1), "yyyy-MM-dd");
    if (rangeOverlapsBooking(tentativeStart, tentativeEnd, bookings)) {
      setCheckIn(dateIso);
      setCheckOut("");
      return;
    }

    setCheckOut(dateIso);
    if (typeof document !== "undefined") {
      setTimeout(() => {
        document
          .getElementById("book")
          ?.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 80);
    }
  }

  return (
    <>
      <AvailabilityCalendar
        bookings={bookings}
        frostStrength={frostStrength}
        selectedCheckIn={checkIn || null}
        selectedCheckOut={checkOut || null}
        onDateClick={handleDateClick}
      />
      <BookingForm
        user={user}
        frostStrength={frostStrength}
        checkInValue={checkIn}
        checkOutValue={checkOut}
        onCheckInChange={setCheckIn}
        onCheckOutChange={setCheckOut}
      />
    </>
  );
}
