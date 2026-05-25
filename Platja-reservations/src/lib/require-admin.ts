import { cookies } from "next/headers";
import { redirect } from "next/navigation";
import { ADMIN_COOKIE_NAME, verifySessionCookie } from "@/lib/admin-auth";
import { getCurrentUser } from "@/lib/user";

/**
 * Server-side guard for /admin/* pages. Admin access is two-factor:
 * the visitor must be signed in to a regular user account *and*
 * have a valid admin session. If either is missing they're bounced
 * — to /login first, then /admin/login.
 */
export async function requireAdmin(currentPath = "/admin"): Promise<void> {
  const user = await getCurrentUser();
  if (!user) {
    redirect(`/login?next=${encodeURIComponent(currentPath)}`);
  }
  const cookie = cookies().get(ADMIN_COOKIE_NAME)?.value;
  const ok = await verifySessionCookie(cookie);
  if (!ok) {
    redirect(`/admin/login?next=${encodeURIComponent(currentPath)}`);
  }
}
