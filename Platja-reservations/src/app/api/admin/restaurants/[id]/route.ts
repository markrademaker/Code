import { NextResponse } from "next/server";
import { cookies } from "next/headers";
import { z } from "zod";
import { prisma } from "@/lib/db";
import { ADMIN_COOKIE_NAME, verifySessionCookie } from "@/lib/admin-auth";

export const runtime = "nodejs";

const schema = z.object({
  name: z.string().min(1).max(120).optional(),
  website: z.string().max(300).optional().nullable(),
  phone: z.string().max(40).optional().nullable(),
  area: z.string().max(120).optional().nullable(),
  blurb: z.string().max(1000).optional().nullable(),
  sortOrder: z.coerce.number().int().optional(),
});

async function requireAuth(): Promise<boolean> {
  return verifySessionCookie(cookies().get(ADMIN_COOKIE_NAME)?.value);
}

export async function PATCH(req: Request, { params }: { params: { id: string } }) {
  if (!(await requireAuth())) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  const body = await req.json().catch(() => null);
  const parsed = schema.safeParse(body);
  if (!parsed.success) {
    return NextResponse.json({ error: "Invalid input" }, { status: 400 });
  }
  const data: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(parsed.data)) {
    if (v === undefined) continue;
    data[k] = v === "" ? null : v;
  }
  try {
    await prisma.restaurant.update({ where: { id: params.id }, data });
  } catch {
    return NextResponse.json({ error: "Not found" }, { status: 404 });
  }
  return NextResponse.json({ ok: true });
}

export async function DELETE(_req: Request, { params }: { params: { id: string } }) {
  if (!(await requireAuth())) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  try {
    await prisma.restaurant.delete({ where: { id: params.id } });
  } catch {
    return NextResponse.json({ error: "Not found" }, { status: 404 });
  }
  return NextResponse.json({ ok: true });
}
