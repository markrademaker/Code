import { NextResponse } from "next/server";
import { z } from "zod";
import { prisma } from "@/lib/db";
import { requireAdminApi } from "@/lib/api-admin";
import { sendMessageToGuest } from "@/lib/email";

export const runtime = "nodejs";

const schema = z.object({ body: z.string().min(1).max(2000) });

export async function GET(_req: Request, { params }: { params: { id: string } }) {
  const auth = await requireAdminApi();
  if (!auth.ok) return auth.response;

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
  const auth = await requireAdminApi();
  if (!auth.ok) return auth.response;

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

  try {
    await sendMessageToGuest(booking, parsed.data.body);
  } catch {
    // ignore email failure — the message is saved
  }

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
