import { NextResponse } from "next/server";
import { prisma } from "@/lib/db";
import { getCurrentUser } from "@/lib/user";

export const runtime = "nodejs";

export async function POST(_req: Request, { params }: { params: { id: string } }) {
  const user = await getCurrentUser();
  if (!user) return NextResponse.json({ error: "Not signed in" }, { status: 401 });

  const booking = await prisma.booking.findUnique({ where: { id: params.id } });
  if (!booking || booking.userId !== user.id) {
    return NextResponse.json({ error: "Not found" }, { status: 404 });
  }
  if (booking.paymentStatus === "PAID") {
    return NextResponse.json({ ok: true });
  }
  await prisma.booking.update({
    where: { id: booking.id },
    data: { paymentStatus: "AWAITING_VERIFICATION" },
  });
  return NextResponse.json({ ok: true });
}
