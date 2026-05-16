import { createHmac, timingSafeEqual } from "node:crypto";

export type BookingAction = "accept" | "tentative" | "decline";

const VALID_ACTIONS: BookingAction[] = ["accept", "tentative", "decline"];

function getSecret(): string {
  const secret = process.env.BOOKING_ACTION_SECRET;
  if (!secret) throw new Error("BOOKING_ACTION_SECRET is not configured");
  return secret;
}

function b64url(buf: Buffer): string {
  return buf.toString("base64").replace(/=/g, "").replace(/\+/g, "-").replace(/\//g, "_");
}

function b64urlDecode(s: string): Buffer {
  const padded = s.replace(/-/g, "+").replace(/_/g, "/") + "===".slice((s.length + 3) % 4);
  return Buffer.from(padded, "base64");
}

function sign(payload: string): string {
  return b64url(createHmac("sha256", getSecret()).update(payload).digest());
}

export function signActionToken(
  bookingId: string,
  action: BookingAction,
  ttlSeconds = 60 * 60 * 24 * 14,
): string {
  const exp = Math.floor(Date.now() / 1000) + ttlSeconds;
  const payload = b64url(Buffer.from(JSON.stringify({ b: bookingId, a: action, e: exp })));
  return `${payload}.${sign(payload)}`;
}

export type DecodedToken = {
  bookingId: string;
  action: BookingAction;
  expiresAt: number;
};

export function verifyActionToken(token: string): DecodedToken | null {
  const dot = token.lastIndexOf(".");
  if (dot < 0) return null;
  const payload = token.slice(0, dot);
  const signature = token.slice(dot + 1);

  const expected = sign(payload);
  if (expected.length !== signature.length) return null;
  try {
    if (!timingSafeEqual(Buffer.from(signature), Buffer.from(expected))) return null;
  } catch {
    return null;
  }

  let decoded: { b?: unknown; a?: unknown; e?: unknown };
  try {
    decoded = JSON.parse(b64urlDecode(payload).toString("utf8"));
  } catch {
    return null;
  }

  if (typeof decoded.b !== "string" || typeof decoded.a !== "string" || typeof decoded.e !== "number") {
    return null;
  }
  if (!VALID_ACTIONS.includes(decoded.a as BookingAction)) return null;
  if (decoded.e < Math.floor(Date.now() / 1000)) return null;

  return { bookingId: decoded.b, action: decoded.a as BookingAction, expiresAt: decoded.e };
}
