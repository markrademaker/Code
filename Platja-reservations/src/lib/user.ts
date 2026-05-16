import { cookies } from "next/headers";
import { prisma } from "@/lib/db";
import { USER_COOKIE_NAME, verifyUserSession } from "@/lib/user-auth";

export type CurrentUser = {
  id: string;
  email: string;
  name: string;
  phone: string | null;
};

export async function getCurrentUser(): Promise<CurrentUser | null> {
  const cookie = cookies().get(USER_COOKIE_NAME)?.value;
  const userId = verifyUserSession(cookie);
  if (!userId) return null;
  try {
    const user = await prisma.user.findUnique({
      where: { id: userId },
      select: { id: true, email: true, name: true, phone: true },
    });
    return user;
  } catch {
    return null;
  }
}
