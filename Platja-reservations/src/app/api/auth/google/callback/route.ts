import { NextResponse } from "next/server";
import { prisma } from "@/lib/db";
import {
  USER_COOKIE_NAME,
  USER_COOKIE_MAX_AGE,
  signUserSession,
} from "@/lib/user-auth";
import { OAUTH_STATE_COOKIE, verifyOAuthState } from "@/lib/oauth-state";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

function getBaseUrl(req: Request): string {
  return (
    process.env.APP_BASE_URL?.replace(/\/$/, "") ?? new URL(req.url).origin
  );
}

function expectedInviteCode(): string {
  return (process.env.SIGNUP_INVITE_CODE ?? "rademaker").trim().toLowerCase();
}

function redirectWithMessage(req: Request, params: Record<string, string>) {
  const url = new URL("/login", getBaseUrl(req));
  for (const [k, v] of Object.entries(params)) url.searchParams.set(k, v);
  const res = NextResponse.redirect(url);
  res.cookies.set({ name: OAUTH_STATE_COOKIE, value: "", path: "/", maxAge: 0 });
  return res;
}

export async function GET(req: Request) {
  const url = new URL(req.url);
  const code = url.searchParams.get("code");
  const state = url.searchParams.get("state");
  const error = url.searchParams.get("error");

  if (error || !code || !state) {
    return redirectWithMessage(req, {
      authError: error ?? "Google sign-in was cancelled",
    });
  }

  const cookies = req.headers.get("cookie") ?? "";
  const stateCookie = cookies
    .split(";")
    .map((c) => c.trim())
    .find((c) => c.startsWith(`${OAUTH_STATE_COOKIE}=`))
    ?.slice(OAUTH_STATE_COOKIE.length + 1);

  const verified = verifyOAuthState(stateCookie, state);
  if (!verified) {
    return redirectWithMessage(req, { authError: "Sign-in expired, please try again" });
  }

  const clientId = process.env.GOOGLE_CLIENT_ID;
  const clientSecret = process.env.GOOGLE_CLIENT_SECRET;
  if (!clientId || !clientSecret) {
    return redirectWithMessage(req, { authError: "Google sign-in is not configured" });
  }

  let accessToken: string;
  try {
    const tokenRes = await fetch("https://oauth2.googleapis.com/token", {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({
        code,
        client_id: clientId,
        client_secret: clientSecret,
        redirect_uri: `${getBaseUrl(req)}/api/auth/google/callback`,
        grant_type: "authorization_code",
      }),
    });
    if (!tokenRes.ok) throw new Error(`Token exchange failed: ${tokenRes.status}`);
    const tokenJson = (await tokenRes.json()) as { access_token?: string };
    if (!tokenJson.access_token) throw new Error("No access_token in response");
    accessToken = tokenJson.access_token;
  } catch (err) {
    console.error("google token exchange failed", err);
    return redirectWithMessage(req, { authError: "Could not reach Google, please try again" });
  }

  let profile: { id: string; email: string; verified_email?: boolean; name?: string };
  try {
    const profRes = await fetch("https://www.googleapis.com/oauth2/v2/userinfo", {
      headers: { Authorization: `Bearer ${accessToken}` },
    });
    if (!profRes.ok) throw new Error(`Userinfo failed: ${profRes.status}`);
    profile = (await profRes.json()) as typeof profile;
  } catch (err) {
    console.error("google profile fetch failed", err);
    return redirectWithMessage(req, { authError: "Could not load your Google profile" });
  }

  if (!profile.email) {
    return redirectWithMessage(req, { authError: "Google didn't share an email address" });
  }
  if (profile.verified_email === false) {
    return redirectWithMessage(req, {
      authError: "Please verify your Google email first",
    });
  }
  const email = profile.email.toLowerCase().trim();
  const googleName = (profile.name ?? email.split("@")[0]).slice(0, 120);

  try {
    let user =
      (await prisma.user.findUnique({ where: { googleId: profile.id } })) ??
      (await prisma.user.findUnique({ where: { email } }));

    if (!user) {
      if (verified.mode !== "signup") {
        return redirectWithMessage(req, {
          authError: "No account for that Google email — sign up first",
          mode: "signup",
        });
      }
      const given = (verified.inviteCode ?? "").trim().toLowerCase();
      if (given !== expectedInviteCode()) {
        return redirectWithMessage(req, {
          authError: "Wrong or missing activation code",
          mode: "signup",
        });
      }
      user = await prisma.user.create({
        data: {
          email,
          googleId: profile.id,
          name: googleName,
        },
      });
    } else if (!user.googleId) {
      user = await prisma.user.update({
        where: { id: user.id },
        data: { googleId: profile.id },
      });
    }

    const sessionCookie = signUserSession(user.id);
    const res = NextResponse.redirect(new URL("/", getBaseUrl(req)));
    res.cookies.set({
      name: USER_COOKIE_NAME,
      value: sessionCookie,
      httpOnly: true,
      secure: process.env.NODE_ENV === "production",
      sameSite: "lax",
      path: "/",
      maxAge: USER_COOKIE_MAX_AGE,
    });
    res.cookies.set({ name: OAUTH_STATE_COOKIE, value: "", path: "/", maxAge: 0 });
    return res;
  } catch (err) {
    console.error("google signin db failed", err);
    return redirectWithMessage(req, { authError: "Could not sign you in, please try again" });
  }
}
