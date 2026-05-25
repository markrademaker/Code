import { NextResponse } from "next/server";
import { z } from "zod";
import { prisma } from "@/lib/db";
import { requireAdminApi } from "@/lib/api-admin";

export const runtime = "nodejs";

const schema = z.object({
  startDate: z.string().regex(/^\d{4}-\d{2}-\d{2}$/).optional(),
  endDate: z.string().regex(/^\d{4}-\d{2}-\d{2}$/).optional(),
  nightlyEuro: z.coerce.number().nonnegative().max(10000).optional(),
  label: z.string().max(120).optional().nullable(),
});

export async function PATCH(req: Request, { params }: { params: { id: string } }) {
  const auth = await requireAdminApi();
  if (!auth.ok) return auth.response;

  const body = await req.json().catch(() => null);
  const parsed = schema.safeParse(body);
  if (!parsed.success) {
    const issues = parsed.error.flatten();
    const firstField = Object.entries(issues.fieldErrors)[0];
    const detail = firstField
      ? `${firstField[0]}: ${firstField[1]?.[0] ?? "invalid"}`
      : (issues.formErrors[0] ?? "invalid");
    return NextResponse.json({ error: `Invalid input — ${detail}`, issues }, { status: 400 });
  }

  const data: Record<string, unknown> = {};
  if (parsed.data.startDate) data.startDate = new Date(parsed.data.startDate);
  if (parsed.data.endDate) data.endDate = new Date(parsed.data.endDate);
  if (parsed.data.nightlyEuro !== undefined)
    data.nightlyRateCents = Math.round(parsed.data.nightlyEuro * 100);
  if (parsed.data.label !== undefined) data.label = parsed.data.label?.trim() || null;

  try {
    await prisma.ratePeriod.update({ where: { id: params.id }, data });
  } catch (err) {
    console.error("rate update failed", err);
    const message = err instanceof Error ? err.message : "Unknown error";
    return NextResponse.json({ error: `Could not save — ${message}` }, { status: 500 });
  }
  return NextResponse.json({ ok: true });
}

export async function DELETE(_req: Request, { params }: { params: { id: string } }) {
  const auth = await requireAdminApi();
  if (!auth.ok) return auth.response;

  try {
    await prisma.ratePeriod.delete({ where: { id: params.id } });
  } catch {
    return NextResponse.json({ error: "Not found" }, { status: 404 });
  }
  return NextResponse.json({ ok: true });
}
