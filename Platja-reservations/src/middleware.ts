import { NextRequest, NextResponse } from "next/server";
import { ADMIN_COOKIE_NAME, verifySessionCookie } from "@/lib/admin-auth";
import { USER_COOKIE_NAME, verifyUserSessionEdge } from "@/lib/user-session-edge";

// Pages crawlers + anonymous visitors can see without signing in.
const PUBLIC_PATHS = new Set<string>([
  "/",
  "/photos",
  "/login",
  "/booking-action",
]);

// Pages that genuinely require a user session.
const REQUIRES_USER_PREFIXES = [
  "/my-bookings",
  "/pay",
  "/weather",
  "/restaurants",
  "/house-rules",
];

export async function middleware(req: NextRequest) {
  const { pathname } = req.nextUrl;

  if (pathname.startsWith("/admin")) {
    // Admin requires two things: a signed-in user account first,
    // then a separate admin password. Stops anyone from typing
    // /admin and going straight to the password prompt.
    const adminUserCookie = req.cookies.get(USER_COOKIE_NAME)?.value;
    const adminUserId = await verifyUserSessionEdge(adminUserCookie);
    if (!adminUserId) {
      const url = req.nextUrl.clone();
      url.pathname = "/login";
      url.searchParams.set("next", pathname);
      return NextResponse.redirect(url);
    }
    if (pathname === "/admin/login") return NextResponse.next();
    const cookie = req.cookies.get(ADMIN_COOKIE_NAME)?.value;
    if (await verifySessionCookie(cookie)) return NextResponse.next();
    const url = req.nextUrl.clone();
    url.pathname = "/admin/login";
    url.searchParams.set("next", pathname);
    return NextResponse.redirect(url);
  }

  if (PUBLIC_PATHS.has(pathname)) return NextResponse.next();

  const needsAuth = REQUIRES_USER_PREFIXES.some(
    (p) => pathname === p || pathname.startsWith(`${p}/`),
  );
  if (!needsAuth) return NextResponse.next();

  const userCookie = req.cookies.get(USER_COOKIE_NAME)?.value;
  const userId = await verifyUserSessionEdge(userCookie);
  if (userId) return NextResponse.next();

  const url = req.nextUrl.clone();
  url.pathname = "/login";
  url.searchParams.set("next", pathname);
  return NextResponse.redirect(url);
}

export const config = {
  matcher: ["/((?!api|_next/static|_next/image|favicon.ico|.*\\.).*)"],
};
