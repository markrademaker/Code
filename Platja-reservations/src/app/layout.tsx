import type { Metadata, Viewport } from "next";
import "./globals.css";
import { NavBar } from "@/components/NavBar";
import { ChatWidget } from "@/components/ChatWidget";
import { getCurrentUser } from "@/lib/user";

export const metadata: Metadata = {
  title: "Villa Mas Nou — Platja d'Aro Reservations",
  description:
    "Reserve our villa in Mas Nou, near Platja d'Aro on the Costa Brava. View availability and request a booking.",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 5,
  themeColor: "#1f4a5f",
};

export default async function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const user = await getCurrentUser();
  const navUser = user ? { name: user.name } : null;
  return (
    <html lang="en">
      <body>
        <NavBar user={navUser} />
        {children}
        {user && <ChatWidget mode="guest" />}
      </body>
    </html>
  );
}
