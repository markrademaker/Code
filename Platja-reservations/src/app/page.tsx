import { Hero } from "@/components/Hero";
import { VillaOverview } from "@/components/VillaOverview";
import { AvailabilityCalendar } from "@/components/AvailabilityCalendar";
import { BookingForm } from "@/components/BookingForm";
import { getBlockingBookings } from "@/lib/bookings";
import { getCurrentUser } from "@/lib/user";

export const dynamic = "force-dynamic";

export default async function HomePage() {
  const [bookings, user] = await Promise.all([
    getBlockingBookings().catch(() => []),
    getCurrentUser(),
  ]);
  return (
    <main>
      <Hero />
      <VillaOverview />
      <AvailabilityCalendar bookings={bookings} />
      <BookingForm user={user} />
      <footer className="border-t border-deep/10 bg-white/60 py-10 text-center text-sm text-deep/60">
        © {new Date().getFullYear()} Villa Mas Nou · Platja d&apos;Aro
      </footer>
    </main>
  );
}
