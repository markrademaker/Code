import { NextResponse } from "next/server";
import { z } from "zod";
import { prisma } from "@/lib/db";
import { isRangeAvailable } from "@/lib/bookings";
import { sendAdminBookingRequest, sendGuestStatusUpdate } from "@/lib/email";

export const runtime = "nodejs";

const schema = z.object({
  name: z.string().min(1).max(120),
  email: z.string().email(),
  phone: z.string().max(40).optional(),
  checkIn: z.string().regex(/^\d{4}-\d{2}-\d{2}$/),
  checkOut: z.string().regex(/^\d{4}-\d{2}-\d{2}$/),
  guests: z.coerce.number().int().min(1).max(20),
  message: z.string().max(2000).optional(),
});

export async function POST(req: Request) {
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

  if (!(await isRangeAvailable(data.checkIn, data.checkOut))) {
    return NextResponse.json(
      { error: "Those dates are not available" },
      { status: 409 },
    );
  }

  const booking = await prisma.booking.create({
    data: {
      name: data.name,
      email: data.email,
      phone: data.phone,
      guests: data.guests,
      checkIn: new Date(data.checkIn),
      checkOut: new Date(data.checkOut),
      message: data.message,
      status: "PENDING",
    },
  });

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
