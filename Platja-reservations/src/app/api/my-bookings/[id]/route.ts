import { NextResponse } from "next/server";
import { prisma } from "@/lib/db";
import { getCurrentUser } from "@/lib/user";
import { sendGuestStatusUpdate } from "@/lib/email";

export const runtime = "nodejs";

export async function POST(_req: Request, { params }: { params: { id: string } }) {
  const user = await getCurrentUser();
  if (!user) return NextResponse.json({ error: "Not signed in" }, { status: 401 });

  const booking = await prisma.booking.findUnique({ where: { id: params.id } });
  if (!booking || booking.userId !== user.id) {
    return NextResponse.json({ error: "Not found" }, { status: 404 });
  }
  if (booking.status === "CANCELLED" || booking.status === "DECLINED") {
    return NextResponse.json({ error: "Already closed" }, { status: 409 });
  }

  const updated = await prisma.booking.update({
    where: { id: booking.id },
    data: { status: "CANCELLED" },
  });

  // Best-effort email; ignore failure
  try {
    await sendGuestStatusUpdate(updated, booking.status);
  } catch {
    // ignore
  }

  return NextResponse.json({ ok: true });
}
