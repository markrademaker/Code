import { NextRequest, NextResponse } from "next/server";
import { ADMIN_COOKIE_NAME, verifySessionCookie } from "@/lib/admin-auth";
import { USER_COOKIE_NAME, verifyUserSessionEdge } from "@/lib/user-session-edge";

const ALWAYS_PUBLIC = new Set<string>(["/login", "/booking-action"]);

export async function middleware(req: NextRequest) {
  const { pathname } = req.nextUrl;

  if (pathname.startsWith("/admin")) {
    if (pathname === "/admin/login") return NextResponse.next();
    const cookie = req.cookies.get(ADMIN_COOKIE_NAME)?.value;
    if (await verifySessionCookie(cookie)) return NextResponse.next();
    const url = req.nextUrl.clone();
    url.pathname = "/admin/login";
    url.searchParams.set("next", pathname);
    return NextResponse.redirect(url);
  }

  if (ALWAYS_PUBLIC.has(pathname)) return NextResponse.next();

  const userCookie = req.cookies.get(USER_COOKIE_NAME)?.value;
  const userId = await verifyUserSessionEdge(userCookie);
  if (userId) return NextResponse.next();

  const url = req.nextUrl.clone();
  url.pathname = "/login";
  if (pathname !== "/") url.searchParams.set("next", pathname);
  return NextResponse.redirect(url);
}

export const config = {
  matcher: ["/((?!api|_next/static|_next/image|favicon.ico|.*\\.).*)"],
};
