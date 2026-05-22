import Link from "next/link";
import { prisma } from "@/lib/db";
import { getCurrentUser } from "@/lib/user";
import { MyBookingsList } from "@/components/MyBookingsList";

export const dynamic = "force-dynamic";

export default async function MyBookingsPage() {
  const user = await getCurrentUser();
  if (!user) {
    // Middleware should have caught this, but just in case
    return (
      <main className="mx-auto max-w-3xl px-5 py-16 sm:px-6">
        <p>Please <Link href="/login" className="text-ocean underline">sign in</Link>.</p>
      </main>
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
    paymentDueDate: b.paymentDueDate ? b.paymentDueDate.toISOString().slice(0, 10) : null,
    createdAt: b.createdAt.toISOString(),
  }));

  return (
    <main className="mx-auto max-w-3xl px-5 py-12 sm:px-6 sm:py-16">
      <p className="text-xs uppercase tracking-[0.2em] text-ink/55">Account</p>
      <h1 className="mt-2 font-display text-3xl font-semibold sm:text-4xl">
        My bookings
      </h1>
      <p className="mt-3 text-base text-ink/70 sm:text-lg">
        Every request you&apos;ve made and where it stands. You can cancel a
        booking that isn&apos;t closed yet.
      </p>

      <div className="mt-8">
        <MyBookingsList bookings={serialized} />
      </div>

      <p className="mt-10 text-sm">
        <Link href="/#book" className="text-ocean hover:underline">
          ← Request another booking
        </Link>
      </p>
    </main>
  );
}
