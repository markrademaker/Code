import type { Metadata } from "next";
import { TodayAtVilla } from "@/components/TodayAtVilla";
import { VillaOverview } from "@/components/VillaOverview";
import { Nearby } from "@/components/Nearby";
import { AvailabilityCalendar } from "@/components/AvailabilityCalendar";
import { BookingForm } from "@/components/BookingForm";
import { Divider } from "@/components/Marks";
import { getBlockingBookings } from "@/lib/bookings";
import { getCurrentUser } from "@/lib/user";

export const metadata: Metadata = {
  title: "Villa Mas Nou — Platja d'Aro, Costa Brava",
  description:
    "Whitewashed family villa with a private pool above Platja d'Aro on the Costa Brava. Sleeps eight, golf next door, the cove five minutes away.",
  alternates: { canonical: "/" },
};

const lodgingJsonLd = {
  "@context": "https://schema.org",
  "@type": "LodgingBusiness",
  "name": "Villa Mas Nou",
  "description":
    "Whitewashed family villa above Platja d'Aro on the Costa Brava. Sleeps eight in four bedrooms with a private pool, sea views, and the Mas Nou golf course next door.",
  "address": {
    "@type": "PostalAddress",
    "addressLocality": "Platja d'Aro",
    "addressRegion": "Catalonia",
    "addressCountry": "ES",
  },
  "geo": {
    "@type": "GeoCoordinates",
    "latitude": 41.817,
    "longitude": 3.067,
  },
  "amenityFeature": [
    { "@type": "LocationFeatureSpecification", "name": "Private pool", "value": true },
    { "@type": "LocationFeatureSpecification", "name": "Sea views", "value": true },
    { "@type": "LocationFeatureSpecification", "name": "Wi-Fi", "value": true },
    { "@type": "LocationFeatureSpecification", "name": "Air conditioning", "value": true },
    { "@type": "LocationFeatureSpecification", "name": "Barbecue", "value": true },
    { "@type": "LocationFeatureSpecification", "name": "Pets allowed", "value": true },
  ],
  "numberOfRooms": 4,
  "petsAllowed": true,
  "smokingAllowed": false,
};

export const dynamic = "force-dynamic";

export default async function HomePage() {
  const [bookings, user] = await Promise.all([
    getBlockingBookings().catch(() => []),
    getCurrentUser(),
  ]);
  return (
    <main>
      <script
        type="application/ld+json"
        // eslint-disable-next-line react/no-danger
        dangerouslySetInnerHTML={{ __html: JSON.stringify(lodgingJsonLd) }}
      />
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
