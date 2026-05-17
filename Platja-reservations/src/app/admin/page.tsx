import { getAllBookings } from "@/lib/bookings";
import { AdminDashboard } from "@/components/AdminDashboard";

export const dynamic = "force-dynamic";

export default async function AdminPage() {
  const bookings = await getAllBookings().catch(() => []);
  const serialized = bookings.map((b) => ({
    id: b.id,
    name: b.name,
    email: b.email,
    phone: b.phone,
    guests: b.guests,
    guestNames: b.guestNames,
    checkIn: b.checkIn.toISOString().slice(0, 10),
    checkOut: b.checkOut.toISOString().slice(0, 10),
    message: b.message,
    status: b.status,
    notes: b.notes,
    ownerNote: b.ownerNote,
    createdAt: b.createdAt.toISOString(),
  }));
  return <AdminDashboard bookings={serialized} />;
}
