import Link from "next/link";
import { format, parseISO } from "date-fns";
import { prisma } from "@/lib/db";
import { getCurrentUser } from "@/lib/user";
import { formatEuro } from "@/lib/pricing";
import { isStripeConfigured } from "@/lib/stripe";
import { PayBookingActions } from "@/components/PayBookingActions";

export const dynamic = "force-dynamic";

export default async function PayPage({ params }: { params: { id: string } }) {
  const user = await getCurrentUser();
  if (!user) {
    return (
      <main className="mx-auto max-w-2xl px-5 py-16 sm:px-6">
        <p>
          Please{" "}
          <Link href="/login" className="text-ocean underline">
            sign in
          </Link>
          .
        </p>
      </main>
    );
  }

  const booking = await prisma.booking.findUnique({ where: { id: params.id } });
  if (!booking || booking.userId !== user.id) {
    return (
      <main className="mx-auto max-w-2xl px-5 py-16 sm:px-6">
        <p>Booking not found.</p>
        <Link href="/my-bookings" className="mt-4 inline-block text-ocean underline">
          ← My bookings
        </Link>
      </main>
    );
  }

  const checkIn = booking.checkIn.toISOString().slice(0, 10);
  const checkOut = booking.checkOut.toISOString().slice(0, 10);
  const due = booking.paymentDueDate
    ? booking.paymentDueDate.toISOString().slice(0, 10)
    : null;

  return (
    <main className="mx-auto max-w-2xl px-5 py-10 sm:px-6 sm:py-14">
      <Link href="/my-bookings" className="text-sm text-ocean hover:underline">
        ← Back to my bookings
      </Link>
      <p className="mt-4 text-xs uppercase tracking-[0.2em] text-ink/55">Payment</p>
      <h1 className="mt-1 font-display text-3xl font-semibold sm:text-4xl">
        {booking.totalAmountCents != null
          ? formatEuro(booking.totalAmountCents)
          : "Total to be confirmed"}
      </h1>
      <p className="mt-2 text-base text-ink/70 sm:text-lg">
        {format(parseISO(checkIn), "EEE d MMM yyyy")} →{" "}
        {format(parseISO(checkOut), "EEE d MMM yyyy")}
      </p>

      <section className="mt-8 rounded-3xl bg-white p-6 shadow-soft ring-1 ring-ink/5 sm:p-7">
        <h2 className="font-display text-xl font-semibold">Bank transfer</h2>
        <p className="mt-2 text-sm text-ink/70">
          Pay by transfer to the account below. We&apos;ll mark your booking as
          paid as soon as we see it arrive.
        </p>
        <dl className="mt-5 grid gap-3 text-sm">
          <Row label="Account holder" value="To be added" />
          <Row label="IBAN" value="To be added" />
          <Row label="BIC / SWIFT" value="To be added" />
          <Row
            label="Reference"
            value={`Villa Mas Nou · ${booking.id.slice(0, 8).toUpperCase()}`}
          />
          {booking.totalAmountCents != null && (
            <Row
              label="Amount"
              value={formatEuro(booking.totalAmountCents)}
              strong
            />
          )}
        </dl>
        {due && (
          <p className="mt-5 rounded-2xl bg-sand/60 px-4 py-3 text-sm text-ink/80">
            Please complete the transfer by{" "}
            <strong>{format(parseISO(due), "EEEE d MMMM yyyy")}</strong> — 7 days
            before check-in.
          </p>
        )}
      </section>

      <PayBookingActions
        bookingId={booking.id}
        paymentStatus={booking.paymentStatus}
        hasAmount={booking.totalAmountCents != null && booking.totalAmountCents > 0}
        stripeConfigured={isStripeConfigured()}
      />
    </main>
  );
}

function Row({
  label,
  value,
  strong,
}: {
  label: string;
  value: string;
  strong?: boolean;
}) {
  return (
    <div className="flex items-center justify-between gap-3 border-b border-ink/5 pb-2 last:border-b-0">
      <dt className="text-ink/55">{label}</dt>
      <dd
        className={`font-mono ${strong ? "font-display text-lg font-semibold text-ink" : "text-ink"}`}
      >
        {value}
      </dd>
    </div>
  );
}
