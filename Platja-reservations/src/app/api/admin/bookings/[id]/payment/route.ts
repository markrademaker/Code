import { NextResponse } from "next/server";
import { cookies } from "next/headers";
import { z } from "zod";
import { prisma } from "@/lib/db";
import { ADMIN_COOKIE_NAME, verifySessionCookie } from "@/lib/admin-auth";

export const runtime = "nodejs";

const schema = z.object({
  paymentStatus: z.enum(["UNPAID", "AWAITING_VERIFICATION", "PAID", "REFUNDED"]),
  totalEuro: z.coerce.number().nonnegative().max(1_000_000).optional(),
});

export async function PATCH(req: Request, { params }: { params: { id: string } }) {
  const ok = await verifySessionCookie(cookies().get(ADMIN_COOKIE_NAME)?.value);
  if (!ok) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  const body = await req.json().catch(() => null);
  const parsed = schema.safeParse(body);
  if (!parsed.success) return NextResponse.json({ error: "Invalid input" }, { status: 400 });

  const data: Record<string, unknown> = { paymentStatus: parsed.data.paymentStatus };
  if (parsed.data.paymentStatus === "PAID") data.paidAt = new Date();
  if (parsed.data.totalEuro !== undefined) {
    data.totalAmountCents = Math.round(parsed.data.totalEuro * 100);
  }

  try {
    await prisma.booking.update({ where: { id: params.id }, data });
  } catch {
    return NextResponse.json({ error: "Not found" }, { status: 404 });
  }
  return NextResponse.json({ ok: true });
}
