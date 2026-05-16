import { NextResponse } from "next/server";
import { prisma } from "@/lib/db";
import { verifyActionToken } from "@/lib/tokens";
import { sendGuestStatusUpdate } from "@/lib/email";
import type { BookingStatus } from "@prisma/client";
import { isRangeAvailable } from "@/lib/bookings";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const ACTION_TO_STATUS: Record<string, BookingStatus> = {
  accept: "CONFIRMED",
  tentative: "TENTATIVE",
  decline: "DECLINED",
};

function redirect(req: Request, params: Record<string, string>): NextResponse {
  const url = new URL("/booking-action", req.url);
  for (const [k, v] of Object.entries(params)) url.searchParams.set(k, v);
  return NextResponse.redirect(url);
}

export async function GET(req: Request) {
  const token = new URL(req.url).searchParams.get("t");
  if (!token) return redirect(req, { state: "invalid" });

  const decoded = verifyActionToken(token);
  if (!decoded) return redirect(req, { state: "invalid" });

  const booking = await prisma.booking.findUnique({ where: { id: decoded.bookingId } });
  if (!booking) return redirect(req, { state: "notfound" });

  const targetStatus = ACTION_TO_STATUS[decoded.action];
  if (!targetStatus) return redirect(req, { state: "invalid" });

  if (booking.status !== "PENDING" && booking.status !== "TENTATIVE") {
    return redirect(req, {
      state: "already",
      status: booking.status,
      guest: booking.name,
    });
  }

  if (targetStatus === "CONFIRMED" || targetStatus === "TENTATIVE") {
    const ok = await isRangeAvailable(
      booking.checkIn.toISOString().slice(0, 10),
      booking.checkOut.toISOString().slice(0, 10),
      booking.id,
    );
    if (!ok) return redirect(req, { state: "conflict", guest: booking.name });
  }

  const updated = await prisma.booking.update({
    where: { id: booking.id },
    data: { status: targetStatus },
  });

  try {
    await sendGuestStatusUpdate(updated, booking.status);
  } catch {
    // status changed, but guest email failed — still show success
  }

  return redirect(req, {
    state: "ok",
    status: updated.status,
    guest: updated.name,
  });
}
