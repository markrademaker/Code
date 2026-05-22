import { prisma } from "@/lib/db";
import { addDays, differenceInCalendarDays, parseISO } from "date-fns";

export type QuoteNight = {
  date: string;
  rateCents: number | null;
};

export type Quote = {
  nights: number;
  coveredNights: number;
  missingNights: number;
  totalCents: number | null;
  currency: "EUR";
  breakdown: QuoteNight[];
};

function toIsoDate(d: Date): string {
  return d.toISOString().slice(0, 10);
}

export async function calculateQuote(
  checkInIso: string,
  checkOutIso: string,
): Promise<Quote | null> {
  const checkIn = parseISO(checkInIso);
  const checkOut = parseISO(checkOutIso);
  const nights = differenceInCalendarDays(checkOut, checkIn);
  if (nights <= 0) return null;

  const periods = await prisma.ratePeriod
    .findMany({
      where: {
        startDate: { lt: checkOut },
        endDate: { gte: checkIn },
      },
      orderBy: { startDate: "asc" },
    })
    .catch(() => []);

  const breakdown: QuoteNight[] = [];
  let total = 0;
  let covered = 0;

  for (let i = 0; i < nights; i++) {
    const day = addDays(checkIn, i);
    const match = periods.find(
      (p) => day >= p.startDate && day <= p.endDate,
    );
    if (match) {
      breakdown.push({ date: toIsoDate(day), rateCents: match.nightlyRateCents });
      total += match.nightlyRateCents;
      covered += 1;
    } else {
      breakdown.push({ date: toIsoDate(day), rateCents: null });
    }
  }

  return {
    nights,
    coveredNights: covered,
    missingNights: nights - covered,
    totalCents: covered === nights ? total : null,
    currency: "EUR",
    breakdown,
  };
}

export function formatEuro(cents: number): string {
  return new Intl.NumberFormat("nl-NL", {
    style: "currency",
    currency: "EUR",
    minimumFractionDigits: 0,
  }).format(cents / 100);
}

export function paymentDueFromCheckIn(checkInIso: string): Date {
  const d = parseISO(checkInIso);
  return addDays(d, -7);
}
