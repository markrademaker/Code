"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useState } from "react";

const LINKS = [
  { href: "/", label: "Villa" },
  { href: "/photos", label: "Photos" },
  { href: "/weather", label: "Weather" },
];

export type NavUser = { name: string } | null;

export function NavBar({ user }: { user: NavUser }) {
  const pathname = usePathname();
  const router = useRouter();
  const [open, setOpen] = useState(false);

  async function logout() {
    await fetch("/api/auth/logout", { method: "POST" });
    setOpen(false);
    router.refresh();
  }

  return (
    <header className="sticky top-0 z-30 border-b border-deep/10 bg-sand/90 backdrop-blur">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-5 py-3 sm:px-6">
        <Link href="/" className="font-display text-lg font-semibold text-deep">
          Villa Mas Nou
        </Link>

        <nav className="hidden items-center gap-1 sm:flex">
          {LINKS.map((l) => {
            const active = pathname === l.href;
            return (
              <Link
                key={l.href}
                href={l.href}
                className={`rounded-full px-4 py-2 text-sm font-medium transition ${
                  active ? "bg-deep text-white" : "text-deep hover:bg-deep/10"
                }`}
              >
                {l.label}
              </Link>
            );
          })}
          {user ? (
            <>
              <span className="ml-2 text-sm text-deep/70">Hi, {user.name.split(" ")[0]}</span>
              <button
                onClick={logout}
                className="rounded-full px-3 py-2 text-sm text-deep/60 hover:bg-deep/5"
              >
                Sign out
              </button>
            </>
          ) : (
            <Link
              href="/login"
              className="rounded-full px-4 py-2 text-sm font-medium text-deep hover:bg-deep/10"
            >
              Sign in
            </Link>
          )}
          <Link
            href="/#book"
            className="ml-1 rounded-full bg-terracotta px-4 py-2 text-sm font-medium text-white hover:bg-terracotta/90"
          >
            Book
          </Link>
        </nav>

        <button
          aria-label="Open menu"
          aria-expanded={open}
          onClick={() => setOpen((v) => !v)}
          className="flex h-10 w-10 items-center justify-center rounded-full bg-white shadow-sm ring-1 ring-deep/10 sm:hidden"
        >
          <span className="sr-only">Menu</span>
          <div className="space-y-1">
            <span className="block h-0.5 w-5 bg-deep"></span>
            <span className="block h-0.5 w-5 bg-deep"></span>
            <span className="block h-0.5 w-5 bg-deep"></span>
          </div>
        </button>
      </div>

      {open && (
        <nav className="border-t border-deep/10 bg-sand sm:hidden">
          <ul className="mx-auto flex max-w-6xl flex-col px-5 py-2">
            {user && (
              <li className="px-4 py-3 text-sm text-deep/70">
                Signed in as <strong className="text-deep">{user.name}</strong>
              </li>
            )}
            {LINKS.map((l) => {
              const active = pathname === l.href;
              return (
                <li key={l.href}>
                  <Link
                    href={l.href}
                    onClick={() => setOpen(false)}
                    className={`block rounded-xl px-4 py-3 text-base font-medium ${
                      active ? "bg-deep text-white" : "text-deep hover:bg-deep/10"
                    }`}
                  >
                    {l.label}
                  </Link>
                </li>
              );
            })}
            <li>
              <Link
                href="/#book"
                onClick={() => setOpen(false)}
                className="mt-1 block rounded-xl bg-terracotta px-4 py-3 text-base font-medium text-white"
              >
                Request a booking
              </Link>
            </li>
            <li className="mt-2 border-t border-deep/10 pt-2">
              {user ? (
                <button
                  onClick={logout}
                  className="block w-full rounded-xl px-4 py-3 text-left text-sm font-medium text-deep/70 hover:bg-deep/5"
                >
                  Sign out
                </button>
              ) : (
                <>
                  <Link
                    href="/login"
                    onClick={() => setOpen(false)}
                    className="block rounded-xl px-4 py-3 text-sm font-medium text-deep/80 hover:bg-deep/5"
                  >
                    Sign in / Create account
                  </Link>
                </>
              )}
              <Link
                href="/admin"
                onClick={() => setOpen(false)}
                className="block rounded-xl px-4 py-3 text-xs font-medium text-deep/50 hover:bg-deep/5"
              >
                Admin sign in
              </Link>
            </li>
          </ul>
        </nav>
      )}
    </header>
  );
}
