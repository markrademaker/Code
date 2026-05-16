import { NextResponse } from "next/server";
import { USER_COOKIE_NAME } from "@/lib/user-auth";

export const runtime = "nodejs";

export async function POST() {
  const res = NextResponse.json({ ok: true });
  res.cookies.set({ name: USER_COOKIE_NAME, value: "", path: "/", maxAge: 0 });
  return res;
}
