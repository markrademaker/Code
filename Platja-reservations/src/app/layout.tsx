import type { Metadata, Viewport } from "next";
import { Cormorant_Garamond, Inter } from "next/font/google";
import "./globals.css";
import { NavBar } from "@/components/NavBar";
import { ChatWidget } from "@/components/ChatWidget";
import { getCurrentUser } from "@/lib/user";

const display = Cormorant_Garamond({
  subsets: ["latin"],
  weight: ["300", "400", "500", "600"],
  style: ["normal", "italic"],
  variable: "--font-display",
  display: "swap",
});

const sans = Inter({
  subsets: ["latin"],
  weight: ["300", "400", "500", "600"],
  variable: "--font-sans",
  display: "swap",
});

function getBaseUrl(): string {
  return (
    process.env.APP_BASE_URL?.replace(/\/$/, "") ??
    process.env.NEXT_PUBLIC_APP_BASE_URL?.replace(/\/$/, "") ??
    "https://platja-reservations.vercel.app"
  );
}

export const metadata: Metadata = {
  metadataBase: new URL(getBaseUrl()),
  title: {
    default: "Villa Mas Nou — Platja d'Aro, Costa Brava",
    template: "%s · Villa Mas Nou",
  },
  description:
    "Whitewashed family villa with a private pool above Platja d'Aro on the Costa Brava. Sleeps eight in four bedrooms, golf next door, the cove five minutes away.",
  keywords: [
    "Villa Mas Nou",
    "Platja d'Aro",
    "Costa Brava villa",
    "villa rental Spain",
    "Mas Nou holiday rental",
    "Catalonia villa",
    "private pool villa Costa Brava",
  ],
  authors: [{ name: "Villa Mas Nou" }],
  openGraph: {
    type: "website",
    locale: "en_GB",
    url: getBaseUrl(),
    siteName: "Villa Mas Nou",
    title: "Villa Mas Nou — Platja d'Aro, Costa Brava",
    description:
      "Whitewashed family villa with a private pool above Platja d'Aro. Sleeps eight, golf next door, the cove five minutes away.",
  },
  twitter: {
    card: "summary_large_image",
    title: "Villa Mas Nou — Platja d'Aro, Costa Brava",
    description:
      "Whitewashed family villa with a private pool above Platja d'Aro.",
  },
  alternates: {
    canonical: "/",
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      "max-image-preview": "large",
      "max-snippet": -1,
    },
  },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 5,
  themeColor: "#3d2f24",
};

export default async function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const user = await getCurrentUser();
  const navUser = user ? { name: user.name } : null;
  return (
    <html lang="en" className={`${display.variable} ${sans.variable}`}>
      <body>
        <NavBar user={navUser} isAdmin={user?.isAdmin ?? false} />
        {children}
        {user && <ChatWidget mode="guest" />}
      </body>
    </html>
  );
}
