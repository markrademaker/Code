import { NextResponse } from "next/server";
import {
  ADMIN_COOKIE_MAX_AGE,
  ADMIN_COOKIE_NAME,
  checkPassword,
  signSessionCookie,
} from "@/lib/admin-auth";

export const runtime = "nodejs";

export async function POST(req: Request) {
  const body = (await req.json().catch(() => null)) as
    | { password?: string; next?: string }
    | null;
  if (!body?.password || !checkPassword(body.password)) {
    return NextResponse.json({ error: "Wrong password" }, { status: 401 });
  }
  const cookie = await signSessionCookie();
  const redirect = body.next?.startsWith("/admin") ? body.next : "/admin";
  const res = NextResponse.json({ ok: true, redirect });
  res.cookies.set({
    name: ADMIN_COOKIE_NAME,
    value: cookie,
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "lax",
    path: "/",
    maxAge: ADMIN_COOKIE_MAX_AGE,
  });
  return res;
}
