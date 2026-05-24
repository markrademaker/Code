import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Booking action",
  robots: { index: false, follow: false },
};

export default function BookingActionLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return children;
}
