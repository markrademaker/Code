import Link from "next/link";
import { prisma } from "@/lib/db";
import { getCurrentUser } from "@/lib/user";
import { MyBookingsList } from "@/components/MyBookingsList";
import { SectionMark } from "@/components/Marks";

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
    <main className="mx-auto max-w-7xl px-5 pt-16 pb-20 sm:px-8 sm:pt-24 sm:pb-28">
      <SectionMark number="IX" label="Account" />

      <div className="mt-10 grid gap-10 lg:grid-cols-[1.45fr_1fr] lg:items-end lg:gap-20">
        <h1 className="font-display text-5xl font-semibold leading-[1.02] tracking-tight text-ink sm:text-7xl lg:text-[5.5rem]">
          My
          <br />
          <span className="italic text-terracotta">bookings</span>.
        </h1>
        <p className="text-base leading-relaxed text-ink/75 sm:text-lg lg:pb-3">
          Every request you&apos;ve made and where it stands. You can cancel a
          booking that isn&apos;t closed yet, message the owners, or pay for
          a confirmed stay.
        </p>
      </div>

      <div className="mt-14 lg:max-w-3xl">
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
