import { NextResponse } from "next/server";
import { z } from "zod";
import { prisma } from "@/lib/db";
import { isRangeAvailable } from "@/lib/bookings";
import { sendAdminBookingRequest, sendGuestStatusUpdate } from "@/lib/email";
import { getCurrentUser } from "@/lib/user";

export const runtime = "nodejs";

const schema = z.object({
  checkIn: z.string().regex(/^\d{4}-\d{2}-\d{2}$/),
  checkOut: z.string().regex(/^\d{4}-\d{2}-\d{2}$/),
  guests: z.coerce.number().int().min(1).max(20),
  message: z.string().max(2000).optional(),
});

export async function POST(req: Request) {
  const user = await getCurrentUser();
  if (!user) {
    return NextResponse.json(
      { error: "You need to sign in to request a booking" },
      { status: 401 },
    );
  }

  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
  }

  const parsed = schema.safeParse(body);
  if (!parsed.success) {
    return NextResponse.json(
      { error: "Invalid input", issues: parsed.error.flatten() },
      { status: 400 },
    );
  }
  const data = parsed.data;

  try {
    if (!(await isRangeAvailable(data.checkIn, data.checkOut))) {
      return NextResponse.json(
        { error: "Those dates are not available" },
        { status: 409 },
      );
    }
  } catch (err) {
    return NextResponse.json(
      { error: `Database error: ${err instanceof Error ? err.message : "unknown"}` },
      { status: 500 },
    );
  }

  let booking;
  try {
    booking = await prisma.booking.create({
      data: {
        name: user.name,
        email: user.email,
        phone: user.phone,
        guests: data.guests,
        checkIn: new Date(data.checkIn),
        checkOut: new Date(data.checkOut),
        message: data.message,
        status: "PENDING",
        userId: user.id,
      },
    });
  } catch (err) {
    return NextResponse.json(
      { error: `Could not save booking: ${err instanceof Error ? err.message : "unknown"}` },
      { status: 500 },
    );
  }

  try {
    await Promise.all([
      sendAdminBookingRequest(booking),
      sendGuestStatusUpdate(booking, null),
    ]);
  } catch (err) {
    const message = err instanceof Error ? err.message : "Unknown error";
    return NextResponse.json(
      { ok: true, warning: `Booking saved but email failed: ${message}` },
      { status: 202 },
    );
  }

  return NextResponse.json({ ok: true });
}
