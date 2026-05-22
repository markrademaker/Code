import { TodayAtVilla } from "@/components/TodayAtVilla";
import { VillaOverview } from "@/components/VillaOverview";
import { Nearby } from "@/components/Nearby";
import { AvailabilityCalendar } from "@/components/AvailabilityCalendar";
import { BookingForm } from "@/components/BookingForm";
import { Divider } from "@/components/Marks";
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
      <VillaOverview />
      <Divider />
      <AvailabilityCalendar bookings={bookings} />
      <BookingForm user={user} />
      <TodayAtVilla />
      <Nearby />
      <footer className="border-t border-ink/10 bg-whitewash py-16 text-center text-sm text-ink/55">
        <p className="font-display text-2xl italic text-ink/70">Villa Mas Nou</p>
        <p className="mt-2 text-xs uppercase tracking-[0.3em]">Platja d&apos;Aro · Costa Brava</p>
        <p className="mt-6 text-xs">© {new Date().getFullYear()}</p>
      </footer>
    </main>
  );
}
