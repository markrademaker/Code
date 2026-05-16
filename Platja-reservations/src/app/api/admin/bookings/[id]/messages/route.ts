import { NextResponse } from "next/server";
import { cookies } from "next/headers";
import { z } from "zod";
import { prisma } from "@/lib/db";
import { ADMIN_COOKIE_NAME, verifySessionCookie } from "@/lib/admin-auth";

export const runtime = "nodejs";

const schema = z.object({ body: z.string().min(1).max(2000) });

async function requireAuth(): Promise<boolean> {
  return verifySessionCookie(cookies().get(ADMIN_COOKIE_NAME)?.value);
}

export async function GET(_req: Request, { params }: { params: { id: string } }) {
  if (!(await requireAuth())) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  const messages = await prisma.bookingMessage.findMany({
    where: { bookingId: params.id },
    orderBy: { createdAt: "asc" },
  });
  return NextResponse.json({
    messages: messages.map((m) => ({
      id: m.id,
      body: m.body,
      fromOwner: m.fromOwner,
      authorName: m.authorName,
      createdAt: m.createdAt.toISOString(),
    })),
  });
}

export async function POST(req: Request, { params }: { params: { id: string } }) {
  if (!(await requireAuth())) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  const body = await req.json().catch(() => null);
  const parsed = schema.safeParse(body);
  if (!parsed.success) return NextResponse.json({ error: "Invalid input" }, { status: 400 });

  const booking = await prisma.booking.findUnique({ where: { id: params.id } });
  if (!booking) return NextResponse.json({ error: "Not found" }, { status: 404 });

  const m = await prisma.bookingMessage.create({
    data: {
      bookingId: booking.id,
      fromOwner: true,
      authorName: "Villa Mas Nou",
      body: parsed.data.body,
    },
  });
  return NextResponse.json({
    ok: true,
    message: {
      id: m.id,
      body: m.body,
      fromOwner: true,
      authorName: m.authorName,
      createdAt: m.createdAt.toISOString(),
    },
  });
}
