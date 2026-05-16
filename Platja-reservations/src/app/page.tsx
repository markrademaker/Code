import { Hero } from "@/components/Hero";
import { VillaOverview } from "@/components/VillaOverview";
import { AvailabilityCalendar } from "@/components/AvailabilityCalendar";
import { BookingForm } from "@/components/BookingForm";
import { getBookings } from "@/lib/bookings";

export default function HomePage() {
  const bookings = getBookings();
  return (
    <main>
      <Hero />
      <VillaOverview />
      <AvailabilityCalendar bookings={bookings} />
      <BookingForm />
      <footer className="border-t border-deep/10 bg-white/60 py-10 text-center text-sm text-deep/60">
        © {new Date().getFullYear()} Villa Mas Nou · Platja d&apos;Aro
      </footer>
    </main>
  );
}
