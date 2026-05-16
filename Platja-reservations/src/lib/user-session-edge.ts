export const USER_COOKIE_NAME = "platja_user";

function getSecret(): string | null {
  const secret = process.env.ADMIN_SESSION_SECRET;
  return secret ? `${secret}:user` : null;
}

function bufToBase64Url(buf: ArrayBuffer): string {
  const u8 = new Uint8Array(buf);
  let s = "";
  for (const b of u8) s += String.fromCharCode(b);
  return btoa(s).replace(/=/g, "").replace(/\+/g, "-").replace(/\//g, "_");
}

function base64UrlDecode(s: string): string {
  const padded = s.replace(/-/g, "+").replace(/_/g, "/") + "===".slice((s.length + 3) % 4);
  return atob(padded);
}

async function hmac(payload: string, secret: string): Promise<string> {
  const key = await crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"],
  );
  const sig = await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(payload));
  return bufToBase64Url(sig);
}

function timingSafeStringEqual(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}

export async function verifyUserSessionEdge(
  cookie: string | undefined,
): Promise<string | null> {
  if (!cookie) return null;
  const secret = getSecret();
  if (!secret) return null;

  const dot = cookie.indexOf(".");
  if (dot < 0) return null;
  const payload = cookie.slice(0, dot);
  const signature = cookie.slice(dot + 1);

  const expected = await hmac(payload, secret);
  if (!timingSafeStringEqual(signature, expected)) return null;

  let data: { uid?: unknown; exp?: unknown };
  try {
    data = JSON.parse(base64UrlDecode(payload));
  } catch {
    return null;
  }
  if (typeof data.uid !== "string" || typeof data.exp !== "number") return null;
  if (data.exp < Math.floor(Date.now() / 1000)) return null;
  return data.uid;
}
