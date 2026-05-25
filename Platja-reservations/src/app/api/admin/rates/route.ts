import { NextResponse } from "next/server";
import { z } from "zod";
import { prisma } from "@/lib/db";
import { requireAdminApi } from "@/lib/api-admin";

export const runtime = "nodejs";

const schema = z.object({
  startDate: z.string().regex(/^\d{4}-\d{2}-\d{2}$/),
  endDate: z.string().regex(/^\d{4}-\d{2}-\d{2}$/),
  nightlyEuro: z.coerce.number().nonnegative().max(10000),
  label: z.string().max(120).optional().nullable(),
});

export async function POST(req: Request) {
  const auth = await requireAdminApi();
  if (!auth.ok) return auth.response;

  const body = await req.json().catch(() => null);
  const parsed = schema.safeParse(body);
  if (!parsed.success) {
    return NextResponse.json({ error: "Invalid input", issues: parsed.error.flatten() }, { status: 400 });
  }
  if (parsed.data.endDate < parsed.data.startDate) {
    return NextResponse.json({ error: "End date must be on or after start date" }, { status: 400 });
  }
  const created = await prisma.ratePeriod.create({
    data: {
      startDate: new Date(parsed.data.startDate),
      endDate: new Date(parsed.data.endDate),
      nightlyRateCents: Math.round(parsed.data.nightlyEuro * 100),
      label: parsed.data.label?.trim() || null,
    },
  });
  return NextResponse.json({ ok: true, id: created.id });
}
