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
    checkIn: b.checkIn.toISOString().slice(0, 10),
    checkOut: b.checkOut.toISOString().slice(0, 10),
    message: b.message,
    status: b.status,
    notes: b.notes,
    createdAt: b.createdAt.toISOString(),
  }));
  return <AdminDashboard bookings={serialized} />;
}
