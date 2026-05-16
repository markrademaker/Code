import bookingsData from "../../data/bookings.json";
import { parseISO, isWithinInterval, isBefore, isEqual } from "date-fns";

export type Booking = {
  id: string;
  start: string;
  end: string;
  label?: string;
};

export function getBookings(): Booking[] {
  return bookingsData.bookings;
}

export function rangesOverlap(
  aStart: Date,
  aEnd: Date,
  bStart: Date,
  bEnd: Date,
): boolean {
  return !(isBefore(aEnd, bStart) || isBefore(bEnd, aStart));
}

export function isRangeAvailable(checkIn: string, checkOut: string): boolean {
  const start = parseISO(checkIn);
  const end = parseISO(checkOut);
  if (isBefore(end, start) || isEqual(end, start)) return false;

  return getBookings().every((b) => {
    const bStart = parseISO(b.start);
    const bEnd = parseISO(b.end);
    return !rangesOverlap(start, end, bStart, bEnd);
  });
}

export function isDateBooked(date: Date): boolean {
  return getBookings().some((b) =>
    isWithinInterval(date, { start: parseISO(b.start), end: parseISO(b.end) }),
  );
}
