import { NextResponse } from "next/server";
import { cookies } from "next/headers";
import { z } from "zod";
import { prisma } from "@/lib/db";
import { ADMIN_COOKIE_NAME, verifySessionCookie } from "@/lib/admin-auth";

export const runtime = "nodejs";

const schema = z.object({
  startDate: z.string().regex(/^\d{4}-\d{2}-\d{2}$/),
  endDate: z.string().regex(/^\d{4}-\d{2}-\d{2}$/),
  nightlyEuro: z.coerce.number().nonnegative().max(10000),
  label: z.string().max(120).optional().nullable(),
});

async function requireAuth(): Promise<boolean> {
  return verifySessionCookie(cookies().get(ADMIN_COOKIE_NAME)?.value);
}

export async function POST(req: Request) {
  if (!(await requireAuth())) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
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
