import { NextResponse } from "next/server";
import { getCurrentUser, type CurrentUser } from "@/lib/user";

/**
 * For server-side API routes that should only be reachable by an
 * admin. Returns the admin user on success, or a NextResponse
 * (401/403) on failure that the route can return directly.
 */
export async function requireAdminApi(): Promise<
  { ok: true; user: CurrentUser } | { ok: false; response: NextResponse }
> {
  const user = await getCurrentUser();
  if (!user) {
    return {
      ok: false,
      response: NextResponse.json({ error: "Sign in required" }, { status: 401 }),
    };
  }
  if (!user.isAdmin) {
    return {
      ok: false,
      response: NextResponse.json(
        { error: "Admin access required" },
        { status: 403 },
      ),
    };
  }
  return { ok: true, user };
}
