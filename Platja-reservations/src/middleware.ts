import { NextRequest, NextResponse } from "next/server";
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

  // Old admin password page is gone — anyone landing there
  // gets bounced to the dashboard (which then runs requireAdmin).
  if (pathname === "/admin/login") {
    const url = req.nextUrl.clone();
    url.pathname = "/admin";
    return NextResponse.redirect(url);
  }

  if (pathname.startsWith("/admin")) {
    // Admin access is RBAC: must be signed in as a regular user.
    // The page itself (via requireAdmin) checks the email allow-list.
    const userCookie = req.cookies.get(USER_COOKIE_NAME)?.value;
    const userId = await verifyUserSessionEdge(userCookie);
    if (!userId) {
      const url = req.nextUrl.clone();
      url.pathname = "/login";
      url.searchParams.set("next", pathname);
      return NextResponse.redirect(url);
    }
    return NextResponse.next();
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
