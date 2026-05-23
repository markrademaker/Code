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

export const metadata: Metadata = {
  title: "Villa Mas Nou — Platja d'Aro Reservations",
  description:
    "Reserve our villa in Mas Nou, near Platja d'Aro on the Costa Brava. View availability and request a booking.",
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
        <NavBar user={navUser} />
        {children}
        {user && <ChatWidget mode="guest" />}
      </body>
    </html>
  );
}
