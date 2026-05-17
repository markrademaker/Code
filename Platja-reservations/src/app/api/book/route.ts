import { NextResponse } from "next/server";
import { z } from "zod";
import { prisma } from "@/lib/db";
import { isRangeAvailable } from "@/lib/bookings";
import { sendAdminBookingRequest, sendGuestStatusUpdate } from "@/lib/email";
import { getCurrentUser } from "@/lib/user";

export const runtime = "nodejs";

const DATE_RE = /^\d{4}-\d{2}-\d{2}$/;
const MAX_NIGHTS = 90;
const MS_PER_DAY = 24 * 60 * 60 * 1000;

const schema = z
  .object({
    checkIn: z.string().regex(DATE_RE),
    checkOut: z.string().regex(DATE_RE),
    guests: z.coerce.number().int().min(1).max(20),
    message: z.string().max(2000).optional(),
  })
  .superRefine((v, ctx) => {
    const inDate = new Date(v.checkIn);
    const outDate = new Date(v.checkOut);
    if (Number.isNaN(inDate.getTime()) || Number.isNaN(outDate.getTime())) {
      ctx.addIssue({ code: "custom", message: "Invalid date" });
      return;
    }
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    if (inDate < today) {
      ctx.addIssue({ code: "custom", path: ["checkIn"], message: "Check-in is in the past" });
    }
    if (outDate <= inDate) {
      ctx.addIssue({
        code: "custom",
        path: ["checkOut"],
        message: "Check-out must be after check-in",
      });
    }
    const nights = (outDate.getTime() - inDate.getTime()) / MS_PER_DAY;
    if (nights > MAX_NIGHTS) {
      ctx.addIssue({
        code: "custom",
        path: ["checkOut"],
        message: `Maximum stay is ${MAX_NIGHTS} nights`,
      });
    }
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
    console.error("availability check failed", err);
    return NextResponse.json(
      { error: "Could not check availability, please try again" },
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
    console.error("booking create failed", err);
    return NextResponse.json(
      { error: "Could not save the booking, please try again" },
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
