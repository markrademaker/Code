import Link from "next/link";
import { format, parseISO } from "date-fns";
import { prisma } from "@/lib/db";
import { getCurrentUser } from "@/lib/user";
import { formatEuro } from "@/lib/pricing";
import { isStripeConfigured } from "@/lib/stripe";
import { PayBookingActions } from "@/components/PayBookingActions";
import { SectionMark } from "@/components/Marks";
import { SiteShell } from "@/components/SiteShell";
import { Frost } from "@/components/Frost";

export const dynamic = "force-dynamic";

export default async function PayPage({ params }: { params: { id: string } }) {
  const user = await getCurrentUser();
  if (!user) {
    return (
      <SiteShell slideshow={false} showFooter={false}>
        <div className="mx-auto max-w-2xl px-5 py-16 sm:px-6">
          <p>
            Please{" "}
            <Link href="/login" className="text-ocean underline">
              sign in
            </Link>
            .
          </p>
        </div>
      </SiteShell>
    );
  }

  const booking = await prisma.booking.findUnique({ where: { id: params.id } });
  if (!booking || booking.userId !== user.id) {
    return (
      <SiteShell slideshow={false} showFooter={false}>
        <div className="mx-auto max-w-2xl px-5 py-16 sm:px-6">
          <p>Booking not found.</p>
          <Link
            href="/my-bookings"
            className="mt-4 inline-block text-ocean underline"
          >
            ← My bookings
          </Link>
        </div>
      </SiteShell>
    );
  }

  const checkIn = booking.checkIn.toISOString().slice(0, 10);
  const checkOut = booking.checkOut.toISOString().slice(0, 10);
  const due = booking.paymentDueDate
    ? booking.paymentDueDate.toISOString().slice(0, 10)
    : null;

  return (
    <SiteShell slideshowCount={2}>
      <div className="pt-12 pb-24 sm:pt-16 sm:pb-32">
        <section className="relative mx-auto max-w-7xl px-5 sm:px-8">
          <Frost strength={80} className="p-8 sm:p-12 lg:p-16">
            <Link
              href="/my-bookings"
              className="text-sm text-ocean hover:underline"
            >
              ← Back to my bookings
            </Link>
            <div className="mt-6">
              <SectionMark number="X" label="Payment" />
            </div>
            <div className="mt-6 grid gap-6 lg:grid-cols-[1.45fr_1fr] lg:items-end lg:gap-20">
              <h1 className="font-display text-4xl font-light leading-[1.05] tracking-tightish text-ink sm:text-5xl lg:text-7xl">
                {booking.totalAmountCents != null ? (
                  <>
                    <span>Pay</span>{" "}
                    {formatEuro(booking.totalAmountCents)}.
                  </>
                ) : (
                  <>
                    Total{" "}
                    <span>to be</span>
                    <br />
                    confirmed.
                  </>
                )}
              </h1>
              <p className="text-base leading-relaxed text-ink/75 sm:text-lg lg:pb-3">
                {format(parseISO(checkIn), "EEE d MMM yyyy")} →{" "}
                {format(parseISO(checkOut), "EEE d MMM yyyy")}
              </p>
            </div>

            <div className="mt-14 grid gap-10 lg:grid-cols-2 lg:gap-16">
              <section className="rounded-3xl bg-white p-6 shadow-soft ring-1 ring-ink/5 sm:p-7">
                <h2 className="font-display text-xl font-semibold">
                  Bank transfer
                </h2>
                <p className="mt-2 text-sm text-ink/70">
                  Pay by transfer to the account below. We&apos;ll mark your
                  booking as paid as soon as we see it arrive.
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
                    <strong>
                      {format(parseISO(due), "EEEE d MMMM yyyy")}
                    </strong>{" "}
                    — 7 days before check-in.
                  </p>
                )}
              </section>

              <div>
                <PayBookingActions
                  bookingId={booking.id}
                  paymentStatus={booking.paymentStatus}
                  hasAmount={
                    booking.totalAmountCents != null &&
                    booking.totalAmountCents > 0
                  }
                  stripeConfigured={isStripeConfigured()}
                />
              </div>
            </div>
          </Frost>
        </section>
      </div>
    </SiteShell>
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
