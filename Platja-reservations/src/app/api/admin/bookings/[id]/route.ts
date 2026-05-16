import { NextResponse } from "next/server";
import { z } from "zod";
import { prisma } from "@/lib/db";
import { isRangeAvailable } from "@/lib/bookings";
import { sendGuestStatusUpdate } from "@/lib/email";
import type { BookingStatus } from "@prisma/client";
import { ADMIN_COOKIE_NAME, verifySessionCookie } from "@/lib/admin-auth";
import { cookies } from "next/headers";

export const runtime = "nodejs";

const STATUSES: BookingStatus[] = [
  "PENDING",
  "TENTATIVE",
  "CONFIRMED",
  "DECLINED",
  "CANCELLED",
];

const patchSchema = z.object({
  status: z.enum(STATUSES as [BookingStatus, ...BookingStatus[]]).optional(),
  checkIn: z.string().regex(/^\d{4}-\d{2}-\d{2}$/).optional(),
  checkOut: z.string().regex(/^\d{4}-\d{2}-\d{2}$/).optional(),
  notes: z.string().max(2000).nullable().optional(),
  notifyGuest: z.boolean().optional(),
});

async function requireAuth(): Promise<boolean> {
  const cookie = cookies().get(ADMIN_COOKIE_NAME)?.value;
  return verifySessionCookie(cookie);
}

export async function PATCH(req: Request, { params }: { params: { id: string } }) {
  if (!(await requireAuth())) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  const body = await req.json().catch(() => null);
  const parsed = patchSchema.safeParse(body);
  if (!parsed.success) {
    return NextResponse.json({ error: "Invalid input", issues: parsed.error.flatten() }, { status: 400 });
  }

  const current = await prisma.booking.findUnique({ where: { id: params.id } });
  if (!current) return NextResponse.json({ error: "Not found" }, { status: 404 });

  const newCheckIn = parsed.data.checkIn ? new Date(parsed.data.checkIn) : current.checkIn;
  const newCheckOut = parsed.data.checkOut ? new Date(parsed.data.checkOut) : current.checkOut;
  const newStatus = parsed.data.status ?? current.status;

  const willBlock = newStatus === "CONFIRMED" || newStatus === "TENTATIVE";
  if (willBlock) {
    const ok = await isRangeAvailable(
      newCheckIn.toISOString().slice(0, 10),
      newCheckOut.toISOString().slice(0, 10),
      current.id,
    );
    if (!ok) return NextResponse.json({ error: "Date conflict with another booking" }, { status: 409 });
  }

  const updated = await prisma.booking.update({
    where: { id: current.id },
    data: {
      status: newStatus,
      checkIn: newCheckIn,
      checkOut: newCheckOut,
      notes: parsed.data.notes === undefined ? current.notes : parsed.data.notes,
    },
  });

  const statusChanged = updated.status !== current.status;
  const datesChanged =
    updated.checkIn.getTime() !== current.checkIn.getTime() ||
    updated.checkOut.getTime() !== current.checkOut.getTime();

  if ((statusChanged || datesChanged) && parsed.data.notifyGuest !== false) {
    try {
      await sendGuestStatusUpdate(updated, current.status);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unknown error";
      return NextResponse.json({ ok: true, warning: `Saved but guest email failed: ${message}` });
    }
  }

  return NextResponse.json({ ok: true });
}

export async function DELETE(_req: Request, { params }: { params: { id: string } }) {
  if (!(await requireAuth())) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  const existing = await prisma.booking.findUnique({ where: { id: params.id } });
  if (!existing) return NextResponse.json({ error: "Not found" }, { status: 404 });
  await prisma.booking.delete({ where: { id: params.id } });
  return NextResponse.json({ ok: true });
}
