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
    <header className="sticky top-0 z-30 border-b border-ink/5 bg-whitewash/80 backdrop-blur">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-5 py-3.5 sm:px-6">
        <Link href="/" className="font-display text-lg font-semibold text-ink">
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
                  active ? "bg-ink text-whitewash" : "text-ink hover:bg-ink/5"
                }`}
              >
                {l.label}
              </Link>
            );
          })}
          {user ? (
            <>
              <Link
                href="/my-bookings"
                className="ml-2 rounded-full px-3 py-2 text-sm font-medium text-ink hover:bg-ink/5"
              >
                My bookings
              </Link>
              <button
                onClick={logout}
                className="rounded-full px-3 py-2 text-sm text-ink/60 hover:bg-ink/5"
              >
                Sign out
              </button>
            </>
          ) : (
            <Link
              href="/login"
              className="rounded-full px-4 py-2 text-sm font-medium text-ink hover:bg-ink/5"
            >
              Sign in
            </Link>
          )}
          <Link
            href="/#book"
            className="ml-1 rounded-full bg-ocean px-5 py-2 text-sm font-medium text-whitewash shadow-glow hover:bg-ocean/90"
          >
            Book
          </Link>
        </nav>

        <button
          aria-label="Open menu"
          aria-expanded={open}
          onClick={() => setOpen((v) => !v)}
          className="flex h-10 w-10 items-center justify-center rounded-full bg-white shadow-soft ring-1 ring-ink/10 sm:hidden"
        >
          <span className="sr-only">Menu</span>
          <div className="space-y-1">
            <span className="block h-0.5 w-5 bg-ink"></span>
            <span className="block h-0.5 w-5 bg-ink"></span>
            <span className="block h-0.5 w-5 bg-ink"></span>
          </div>
        </button>
      </div>

      {open && (
        <nav className="border-t border-ink/10 bg-whitewash sm:hidden">
          <ul className="mx-auto flex max-w-6xl flex-col px-5 py-2">
            {user && (
              <li className="px-4 py-3 text-sm text-ink/70">
                Signed in as <strong className="text-ink">{user.name}</strong>
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
                      active ? "bg-ink text-whitewash" : "text-ink hover:bg-ink/5"
                    }`}
                  >
                    {l.label}
                  </Link>
                </li>
              );
            })}
            {user && (
              <li>
                <Link
                  href="/my-bookings"
                  onClick={() => setOpen(false)}
                  className="block rounded-xl px-4 py-3 text-base font-medium text-ink hover:bg-ink/5"
                >
                  My bookings
                </Link>
              </li>
            )}
            <li>
              <Link
                href="/#book"
                onClick={() => setOpen(false)}
                className="mt-1 block rounded-xl bg-ocean px-4 py-3 text-base font-medium text-whitewash"
              >
                Request a booking
              </Link>
            </li>
            <li className="mt-2 border-t border-ink/10 pt-2">
              {user ? (
                <button
                  onClick={logout}
                  className="block w-full rounded-xl px-4 py-3 text-left text-sm font-medium text-ink/70 hover:bg-ink/5"
                >
                  Sign out
                </button>
              ) : (
                <Link
                  href="/login"
                  onClick={() => setOpen(false)}
                  className="block rounded-xl px-4 py-3 text-sm font-medium text-ink/80 hover:bg-ink/5"
                >
                  Sign in / Create account
                </Link>
              )}
              <Link
                href="/admin"
                onClick={() => setOpen(false)}
                className="block rounded-xl px-4 py-3 text-xs font-medium text-ink/50 hover:bg-ink/5"
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
