import { Hero } from "@/components/Hero";
import { VillaOverview } from "@/components/VillaOverview";
import { Nearby } from "@/components/Nearby";
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
      <Nearby />
      <AvailabilityCalendar bookings={bookings} />
      <BookingForm user={user} />
      <footer className="border-t border-ink/10 bg-whitewash py-12 text-center text-sm text-ink/55">
        <p className="font-display text-lg text-ink/70">Villa Mas Nou</p>
        <p className="mt-1">Platja d&apos;Aro · Costa Brava</p>
        <p className="mt-3 text-xs">© {new Date().getFullYear()}</p>
      </footer>
    </main>
  );
}
