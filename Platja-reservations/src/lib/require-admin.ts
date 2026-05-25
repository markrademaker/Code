import { redirect } from "next/navigation";
import { getCurrentUser } from "@/lib/user";

/**
 * Server-side guard for /admin/* pages. The visitor must be signed
 * in to a regular user account *and* that user's email must be on
 * the admin allow-list (src/lib/admin-rbac.ts). If they're not
 * signed in we redirect to /login with ?next= back. If they're
 * signed in but not an admin we bounce them home.
 */
export async function requireAdmin(currentPath = "/admin"): Promise<void> {
  const user = await getCurrentUser();
  if (!user) {
    redirect(`/login?next=${encodeURIComponent(currentPath)}`);
  }
  if (!user.isAdmin) {
    redirect("/");
  }
}
