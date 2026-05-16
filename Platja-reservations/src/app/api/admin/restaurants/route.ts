import { NextResponse } from "next/server";
import { cookies } from "next/headers";
import { z } from "zod";
import { prisma } from "@/lib/db";
import { ADMIN_COOKIE_NAME, verifySessionCookie } from "@/lib/admin-auth";

export const runtime = "nodejs";

const schema = z.object({
  name: z.string().min(1).max(120),
  website: z.string().max(300).optional().nullable(),
  phone: z.string().max(40).optional().nullable(),
  area: z.string().max(120).optional().nullable(),
  blurb: z.string().max(1000).optional().nullable(),
  sortOrder: z.coerce.number().int().optional(),
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
  const r = await prisma.restaurant.create({
    data: {
      name: parsed.data.name,
      website: parsed.data.website || null,
      phone: parsed.data.phone || null,
      area: parsed.data.area || null,
      blurb: parsed.data.blurb || null,
      sortOrder: parsed.data.sortOrder ?? 0,
    },
  });
  return NextResponse.json({ ok: true, id: r.id });
}
