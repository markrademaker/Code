import { NextResponse } from "next/server";
import { z } from "zod";
import { prisma } from "@/lib/db";
import { requireAdminApi } from "@/lib/api-admin";

export const runtime = "nodejs";

const schema = z.object({
  paymentStatus: z.enum(["UNPAID", "AWAITING_VERIFICATION", "PAID", "REFUNDED"]),
  totalEuro: z.coerce.number().nonnegative().max(1_000_000).optional(),
});

export async function PATCH(req: Request, { params }: { params: { id: string } }) {
  const auth = await requireAdminApi();
  if (!auth.ok) return auth.response;

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
