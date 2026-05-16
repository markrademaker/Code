import { prisma } from "@/lib/db";
import type { Booking as DbBooking, BookingStatus } from "@prisma/client";
import { isBefore, isEqual, parseISO } from "date-fns";

export type PublicBooking = {
  id: string;
  start: string;
  end: string;
  label?: string;
};

const BLOCKING_STATUSES: BookingStatus[] = ["CONFIRMED", "TENTATIVE"];

function toIsoDate(d: Date): string {
  return d.toISOString().slice(0, 10);
}

export async function getBlockingBookings(): Promise<PublicBooking[]> {
  const rows = await prisma.booking.findMany({
    where: { status: { in: BLOCKING_STATUSES } },
    orderBy: { checkIn: "asc" },
  });
  return rows.map((b) => ({
    id: b.id,
    start: toIsoDate(b.checkIn),
    end: toIsoDate(b.checkOut),
    label: b.status === "TENTATIVE" ? "Tentative" : "Confirmed",
  }));
}

function rangesOverlap(aStart: Date, aEnd: Date, bStart: Date, bEnd: Date): boolean {
  return !(isBefore(aEnd, bStart) || isBefore(bEnd, aStart));
}

export async function isRangeAvailable(
  checkIn: string,
  checkOut: string,
  ignoreBookingId?: string,
): Promise<boolean> {
  const start = parseISO(checkIn);
  const end = parseISO(checkOut);
  if (isBefore(end, start) || isEqual(end, start)) return false;

  const conflicts = await prisma.booking.findMany({
    where: {
      status: { in: BLOCKING_STATUSES },
      ...(ignoreBookingId ? { NOT: { id: ignoreBookingId } } : {}),
    },
  });
  return conflicts.every((b) => !rangesOverlap(start, end, b.checkIn, b.checkOut));
}

export async function getAllBookings(): Promise<DbBooking[]> {
  return prisma.booking.findMany({ orderBy: { checkIn: "asc" } });
}

export function formatDateRange(start: Date, end: Date): string {
  return `${toIsoDate(start)} → ${toIsoDate(end)}`;
}

export { toIsoDate };
