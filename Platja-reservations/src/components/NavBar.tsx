"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";

const LINKS = [
  { href: "/", label: "Villa" },
  { href: "/photos", label: "Photos" },
  { href: "/weather", label: "Weather" },
];

export function NavBar() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);

  return (
    <header className="sticky top-0 z-30 border-b border-deep/10 bg-sand/90 backdrop-blur">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-5 py-3 sm:px-6">
        <Link href="/" className="font-display text-lg font-semibold text-deep">
          Villa Mas Nou
        </Link>

        <nav className="hidden gap-1 sm:flex">
          {LINKS.map((l) => {
            const active = pathname === l.href;
            return (
              <Link
                key={l.href}
                href={l.href}
                className={`rounded-full px-4 py-2 text-sm font-medium transition ${
                  active
                    ? "bg-deep text-white"
                    : "text-deep hover:bg-deep/10"
                }`}
              >
                {l.label}
              </Link>
            );
          })}
          <Link
            href="/#book"
            className="ml-2 rounded-full bg-terracotta px-4 py-2 text-sm font-medium text-white hover:bg-terracotta/90"
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
              <Link
                href="/admin"
                onClick={() => setOpen(false)}
                className="block rounded-xl px-4 py-3 text-sm font-medium text-deep/70 hover:bg-deep/5"
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
