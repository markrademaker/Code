import Link from "next/link";
import { prisma } from "@/lib/db";
import { getCurrentUser } from "@/lib/user";
import { MyBookingsList } from "@/components/MyBookingsList";
import { SectionMark } from "@/components/Marks";
import { SiteShell } from "@/components/SiteShell";
import { Frost } from "@/components/Frost";

export const dynamic = "force-dynamic";

export default async function MyBookingsPage() {
  const user = await getCurrentUser();
  if (!user) {
    return (
      <SiteShell slideshow={false} showFooter={false}>
        <div className="mx-auto max-w-3xl px-5 py-16 sm:px-6">
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

  const bookings = await prisma.booking.findMany({
    where: { userId: user.id },
    orderBy: { checkIn: "desc" },
  });

  const serialized = bookings.map((b) => ({
    id: b.id,
    checkIn: b.checkIn.toISOString().slice(0, 10),
    checkOut: b.checkOut.toISOString().slice(0, 10),
    guests: b.guests,
    guestNames: b.guestNames,
    message: b.message,
    status: b.status,
    ownerNote: b.ownerNote,
    totalAmountCents: b.totalAmountCents,
    paymentStatus: b.paymentStatus,
    paymentDueDate: b.paymentDueDate
      ? b.paymentDueDate.toISOString().slice(0, 10)
      : null,
    createdAt: b.createdAt.toISOString(),
  }));

  return (
    <SiteShell slideshowCount={2}>
      <div className="pt-12 pb-24 sm:pt-16 sm:pb-32">
        <section className="relative mx-auto max-w-7xl px-5 sm:px-8">
          <Frost strength={80} className="p-8 sm:p-12 lg:p-16">
            <SectionMark number="IX" label="Account" />

            <div className="mt-10 grid gap-10 lg:grid-cols-[1.45fr_1fr] lg:items-end lg:gap-20">
              <h1 className="font-display text-4xl font-light leading-[1.05] tracking-tightish text-ink sm:text-5xl lg:text-7xl">
                My
                <br />
                <span className="italic text-terracotta">bookings</span>.
              </h1>
              <p className="text-base leading-relaxed text-ink/75 sm:text-lg lg:pb-3">
                Every request you&apos;ve made and where it stands. You can
                cancel a booking that isn&apos;t closed yet, message the
                owners, or pay for a confirmed stay.
              </p>
            </div>

            <div className="mt-14">
              <MyBookingsList bookings={serialized} />
            </div>

            <p className="mt-10 text-sm">
              <Link href="/#book" className="text-ocean hover:underline">
                ← Request another booking
              </Link>
            </p>
          </Frost>
        </section>
      </div>
    </SiteShell>
  );
}
