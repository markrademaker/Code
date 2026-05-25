import type { Metadata } from "next";
import { SiteShell } from "@/components/SiteShell";
import { Hero } from "@/components/Hero";
import { PullQuote } from "@/components/PullQuote";
import { VillaOverview } from "@/components/VillaOverview";
import { Nearby } from "@/components/Nearby";
import { BookingFlow } from "@/components/BookingFlow";
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

  const FROST = 80;

  return (
    <>
      <script
        type="application/ld+json"
        // eslint-disable-next-line react/no-danger
        dangerouslySetInnerHTML={{ __html: JSON.stringify(lodgingJsonLd) }}
      />
      <SiteShell>
        <Hero />
        <div className="space-y-24 pb-24 sm:space-y-32">
          <VillaOverview frostStrength={FROST} />
          <Divider light />
          <PullQuote />
          <BookingFlow bookings={bookings} user={user} frostStrength={FROST} />
          <Nearby frostStrength={FROST} />
        </div>
      </SiteShell>
    </>
  );
}
