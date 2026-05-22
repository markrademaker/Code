import { NextResponse } from "next/server";
import { calculateQuote } from "@/lib/pricing";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const DATE_RE = /^\d{4}-\d{2}-\d{2}$/;

export async function GET(req: Request) {
  const url = new URL(req.url);
  const checkIn = url.searchParams.get("checkIn") ?? "";
  const checkOut = url.searchParams.get("checkOut") ?? "";
  if (!DATE_RE.test(checkIn) || !DATE_RE.test(checkOut)) {
    return NextResponse.json({ error: "Invalid dates" }, { status: 400 });
  }
  const quote = await calculateQuote(checkIn, checkOut);
  if (!quote) return NextResponse.json({ error: "Invalid range" }, { status: 400 });
  return NextResponse.json(quote);
}
