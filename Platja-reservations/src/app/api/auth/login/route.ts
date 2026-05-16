import { NextResponse } from "next/server";
import { z } from "zod";
import { prisma } from "@/lib/db";
import {
  USER_COOKIE_NAME,
  USER_COOKIE_MAX_AGE,
  signUserSession,
  verifyPassword,
} from "@/lib/user-auth";

export const runtime = "nodejs";

const schema = z.object({
  email: z.string().email().max(200),
  password: z.string().min(1).max(200),
});

export async function POST(req: Request) {
  const body = await req.json().catch(() => null);
  const parsed = schema.safeParse(body);
  if (!parsed.success) {
    return NextResponse.json({ error: "Invalid input" }, { status: 400 });
  }
  const email = parsed.data.email.toLowerCase().trim();

  try {
    const user = await prisma.user.findUnique({ where: { email } });
    if (!user) {
      return NextResponse.json({ error: "Wrong email or password" }, { status: 401 });
    }
    const ok = await verifyPassword(parsed.data.password, user.passwordHash);
    if (!ok) {
      return NextResponse.json({ error: "Wrong email or password" }, { status: 401 });
    }
    const cookie = signUserSession(user.id);
    const res = NextResponse.json({ ok: true });
    res.cookies.set({
      name: USER_COOKIE_NAME,
      value: cookie,
      httpOnly: true,
      secure: process.env.NODE_ENV === "production",
      sameSite: "lax",
      path: "/",
      maxAge: USER_COOKIE_MAX_AGE,
    });
    return res;
  } catch (err) {
    const message = err instanceof Error ? err.message : "unknown";
    return NextResponse.json({ error: `Login failed: ${message}` }, { status: 500 });
  }
}
