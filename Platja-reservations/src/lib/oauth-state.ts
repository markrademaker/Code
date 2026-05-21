import { createHmac, randomBytes, timingSafeEqual } from "node:crypto";

export type OAuthStatePayload = {
  mode: "login" | "signup";
  inviteCode?: string;
  nonce: string;
  exp: number;
};

const TTL_SECONDS = 10 * 60;

function getSecret(): string {
  const secret = process.env.ADMIN_SESSION_SECRET;
  if (!secret) throw new Error("ADMIN_SESSION_SECRET is not configured");
  return `${secret}:oauth`;
}

function sign(payload: string): string {
  return createHmac("sha256", getSecret()).update(payload).digest("base64url");
}

export function signOAuthState(
  data: Omit<OAuthStatePayload, "exp" | "nonce">,
): { cookie: string; state: string } {
  const nonce = randomBytes(16).toString("base64url");
  const full: OAuthStatePayload = {
    ...data,
    nonce,
    exp: Math.floor(Date.now() / 1000) + TTL_SECONDS,
  };
  const payload = Buffer.from(JSON.stringify(full)).toString("base64url");
  return { cookie: `${payload}.${sign(payload)}`, state: nonce };
}

export function verifyOAuthState(
  cookie: string | undefined,
  state: string,
): OAuthStatePayload | null {
  if (!cookie || !state) return null;
  const dot = cookie.indexOf(".");
  if (dot < 0) return null;
  const payload = cookie.slice(0, dot);
  const signature = cookie.slice(dot + 1);
  const expected = sign(payload);
  if (signature.length !== expected.length) return null;
  try {
    if (!timingSafeEqual(Buffer.from(signature), Buffer.from(expected))) return null;
  } catch {
    return null;
  }
  let data: OAuthStatePayload;
  try {
    data = JSON.parse(Buffer.from(payload, "base64url").toString("utf8"));
  } catch {
    return null;
  }
  if (data.exp < Math.floor(Date.now() / 1000)) return null;
  if (data.nonce !== state) return null;
  return data;
}

export const OAUTH_STATE_COOKIE = "platja_oauth_state";
