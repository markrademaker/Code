import { cookies } from "next/headers";
import { redirect } from "next/navigation";
import { ADMIN_COOKIE_NAME, verifySessionCookie } from "@/lib/admin-auth";

/**
 * Server-side guard for /admin/* pages. Call at the top of every
 * admin page. If the admin session cookie is missing or invalid,
 * we redirect to /admin/login with a ?next= so they bounce back
 * after signing in. Layered on top of middleware so a single
 * misconfiguration can't accidentally expose the admin area.
 */
export async function requireAdmin(currentPath = "/admin"): Promise<void> {
  const cookie = cookies().get(ADMIN_COOKIE_NAME)?.value;
  const ok = await verifySessionCookie(cookie);
  if (!ok) {
    redirect(`/admin/login?next=${encodeURIComponent(currentPath)}`);
  }
}
