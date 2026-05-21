import { NextResponse } from "next/server";
import { signOAuthState, OAUTH_STATE_COOKIE } from "@/lib/oauth-state";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

function getBaseUrl(req: Request): string {
  return (
    process.env.APP_BASE_URL?.replace(/\/$/, "") ?? new URL(req.url).origin
  );
}

export async function GET(req: Request) {
  const clientId = process.env.GOOGLE_CLIENT_ID;
  if (!clientId) {
    return NextResponse.json(
      { error: "Google sign-in is not configured" },
      { status: 500 },
    );
  }

  const url = new URL(req.url);
  const modeParam = url.searchParams.get("mode");
  const mode: "login" | "signup" = modeParam === "signup" ? "signup" : "login";
  const inviteCode = url.searchParams.get("code") ?? undefined;

  const { cookie, state } = signOAuthState({ mode, inviteCode });

  const auth = new URL("https://accounts.google.com/o/oauth2/v2/auth");
  auth.searchParams.set("client_id", clientId);
  auth.searchParams.set("redirect_uri", `${getBaseUrl(req)}/api/auth/google/callback`);
  auth.searchParams.set("response_type", "code");
  auth.searchParams.set("scope", "openid email profile");
  auth.searchParams.set("state", state);
  auth.searchParams.set("prompt", "select_account");
  auth.searchParams.set("access_type", "online");

  const res = NextResponse.redirect(auth.toString());
  res.cookies.set({
    name: OAUTH_STATE_COOKIE,
    value: cookie,
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "lax",
    path: "/",
    maxAge: 600,
  });
  return res;
}
