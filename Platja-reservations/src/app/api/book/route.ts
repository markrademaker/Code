import { NextResponse } from "next/server";
import { z } from "zod";
import { isRangeAvailable } from "@/lib/bookings";
import { sendBookingEmail } from "@/lib/email";

const schema = z.object({
  name: z.string().min(1).max(120),
  email: z.string().email(),
  phone: z.string().max(40).optional(),
  checkIn: z.string().regex(/^\d{4}-\d{2}-\d{2}$/),
  checkOut: z.string().regex(/^\d{4}-\d{2}-\d{2}$/),
  guests: z.coerce.number().int().min(1).max(20),
  message: z.string().max(2000).optional(),
});

export async function POST(req: Request) {
  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
  }

  const parsed = schema.safeParse(body);
  if (!parsed.success) {
    return NextResponse.json(
      { error: "Invalid input", issues: parsed.error.flatten() },
      { status: 400 },
    );
  }

  const data = parsed.data;

  if (!isRangeAvailable(data.checkIn, data.checkOut)) {
    return NextResponse.json(
      { error: "Those dates are not available" },
      { status: 409 },
    );
  }

  try {
    await sendBookingEmail(data);
  } catch (err) {
    const message = err instanceof Error ? err.message : "Unknown error";
    return NextResponse.json(
      { error: `Failed to send booking email: ${message}` },
      { status: 500 },
    );
  }

  return NextResponse.json({ ok: true });
}
