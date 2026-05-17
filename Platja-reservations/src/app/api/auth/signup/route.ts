import { NextResponse } from "next/server";
import { z } from "zod";
import { prisma } from "@/lib/db";
import {
  USER_COOKIE_NAME,
  USER_COOKIE_MAX_AGE,
  hashPassword,
  signUserSession,
} from "@/lib/user-auth";

export const runtime = "nodejs";

const schema = z.object({
  name: z.string().min(1).max(120),
  email: z.string().email().max(200),
  phone: z.string().max(40).optional(),
  password: z.string().min(8).max(200),
  inviteCode: z.string().min(1).max(120),
});

function expectedInviteCode(): string {
  return (process.env.SIGNUP_INVITE_CODE ?? "rademaker").trim().toLowerCase();
}

export async function POST(req: Request) {
  const body = await req.json().catch(() => null);
  const parsed = schema.safeParse(body);
  if (!parsed.success) {
    return NextResponse.json(
      { error: "Invalid input", issues: parsed.error.flatten() },
      { status: 400 },
    );
  }
  const { name, email, phone, password, inviteCode } = parsed.data;

  if (inviteCode.trim().toLowerCase() !== expectedInviteCode()) {
    return NextResponse.json(
      { error: "Wrong activation code" },
      { status: 403 },
    );
  }

  const normalizedEmail = email.toLowerCase().trim();

  try {
    const existing = await prisma.user.findUnique({ where: { email: normalizedEmail } });
    if (existing) {
      return NextResponse.json(
        { error: "An account with this email already exists" },
        { status: 409 },
      );
    }
    const passwordHash = await hashPassword(password);
    const user = await prisma.user.create({
      data: { name, email: normalizedEmail, phone: phone || null, passwordHash },
    });
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
    return NextResponse.json({ error: `Sign-up failed: ${message}` }, { status: 500 });
  }
}
