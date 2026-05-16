import { NextResponse } from "next/server";
import { z } from "zod";
import { prisma } from "@/lib/db";
import { getCurrentUser } from "@/lib/user";

export const runtime = "nodejs";

const schema = z.object({ body: z.string().min(1).max(2000) });

export async function GET(_req: Request, { params }: { params: { id: string } }) {
  const user = await getCurrentUser();
  if (!user) return NextResponse.json({ error: "Not signed in" }, { status: 401 });
  const booking = await prisma.booking.findUnique({ where: { id: params.id } });
  if (!booking || booking.userId !== user.id) {
    return NextResponse.json({ error: "Not found" }, { status: 404 });
  }
  const messages = await prisma.bookingMessage.findMany({
    where: { bookingId: booking.id },
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
  const user = await getCurrentUser();
  if (!user) return NextResponse.json({ error: "Not signed in" }, { status: 401 });
  const body = await req.json().catch(() => null);
  const parsed = schema.safeParse(body);
  if (!parsed.success) return NextResponse.json({ error: "Invalid input" }, { status: 400 });

  const booking = await prisma.booking.findUnique({ where: { id: params.id } });
  if (!booking || booking.userId !== user.id) {
    return NextResponse.json({ error: "Not found" }, { status: 404 });
  }

  const m = await prisma.bookingMessage.create({
    data: {
      bookingId: booking.id,
      authorId: user.id,
      authorName: user.name,
      fromOwner: false,
      body: parsed.data.body,
    },
  });

  return NextResponse.json({
    ok: true,
    message: {
      id: m.id,
      body: m.body,
      fromOwner: false,
      authorName: m.authorName,
      createdAt: m.createdAt.toISOString(),
    },
  });
}
