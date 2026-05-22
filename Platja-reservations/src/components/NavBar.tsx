"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useState } from "react";
import { SunMark } from "@/components/Marks";

const LINKS = [
  { href: "/", label: "Villa" },
  { href: "/photos", label: "Photos" },
  { href: "/weather", label: "Weather" },
  { href: "/restaurants", label: "Eat" },
  { href: "/house-rules", label: "House rules" },
];

export type NavUser = { name: string } | null;

export function NavBar({ user }: { user: NavUser }) {
  const pathname = usePathname();
  const router = useRouter();
  const [open, setOpen] = useState(false);

  if (pathname === "/login") return null;

  async function logout() {
    await fetch("/api/auth/logout", { method: "POST" });
    setOpen(false);
    router.refresh();
  }

  return (
    <header className="sticky top-0 z-30 border-b border-ink/10 bg-whitewash/85 backdrop-blur">
      <div className="mx-auto flex max-w-7xl items-center justify-between gap-6 px-5 py-4 sm:px-8 sm:py-5">
        <Link href="/" className="group flex items-baseline gap-2.5">
          <span className="text-terracotta transition group-hover:rotate-45">
            <SunMark className="h-4 w-4" />
          </span>
          <span className="font-display text-xl font-semibold italic text-ink sm:text-2xl">
            Villa Mas Nou
          </span>
        </Link>

        <nav className="hidden items-center gap-7 lg:flex">
          {LINKS.map((l) => {
            const active = pathname === l.href;
            return (
              <Link
                key={l.href}
                href={l.href}
                className={`text-[11px] uppercase tracking-[0.2em] transition ${
                  active ? "text-ink" : "text-ink/55 hover:text-ink"
                }`}
              >
                {l.label}
              </Link>
            );
          })}
        </nav>

        <div className="hidden items-center gap-3 lg:flex">
          {user ? (
            <>
              <Link
                href="/my-bookings"
                className="text-[11px] uppercase tracking-[0.2em] text-ink/55 hover:text-ink"
              >
                My bookings
              </Link>
              <button
                onClick={logout}
                className="text-[11px] uppercase tracking-[0.2em] text-ink/45 hover:text-ink"
              >
                Sign out
              </button>
            </>
          ) : (
            <Link
              href="/login"
              className="text-[11px] uppercase tracking-[0.2em] text-ink/55 hover:text-ink"
            >
              Sign in
            </Link>
          )}
          <Link
            href="/#book"
            className="ml-2 rounded-full bg-ocean px-5 py-2 text-xs font-medium uppercase tracking-wider text-whitewash shadow-glow hover:bg-ocean/90"
          >
            Reserve
          </Link>
          <Link
            href="/admin"
            className="text-[10px] uppercase tracking-[0.2em] text-ink/40 hover:text-ink/70"
            title="Admin sign in"
          >
            Admin
          </Link>
        </div>

        <button
          aria-label="Open menu"
          aria-expanded={open}
          onClick={() => setOpen((v) => !v)}
          className="flex h-10 w-10 items-center justify-center rounded-full bg-white shadow-soft ring-1 ring-ink/10 lg:hidden"
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
        <nav className="border-t border-ink/10 bg-whitewash lg:hidden">
          <ul className="mx-auto flex max-w-7xl flex-col px-5 py-2 sm:px-8">
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
