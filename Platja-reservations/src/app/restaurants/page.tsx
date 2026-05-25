import type { Metadata } from "next";
import Link from "next/link";
import { prisma } from "@/lib/db";
import { SectionMark } from "@/components/Marks";
import { SiteShell } from "@/components/SiteShell";
import { Frost } from "@/components/Frost";

export const metadata: Metadata = {
  title: "Where to eat around Platja d'Aro",
  description:
    "Our favourite restaurants around Platja d'Aro and the Costa Brava — tapas, seafood, sunset dinners, and Sunday lunches.",
  alternates: { canonical: "/restaurants" },
};

export const dynamic = "force-dynamic";

function normalizeUrl(raw: string): string {
  if (/^https?:\/\//i.test(raw)) return raw;
  return `https://${raw}`;
}

function siteHost(raw: string): string {
  try {
    return new URL(normalizeUrl(raw)).host.replace(/^www\./, "");
  } catch {
    return raw;
  }
}

export default async function RestaurantsPage() {
  const restaurants = await prisma.restaurant
    .findMany({ orderBy: [{ sortOrder: "asc" }, { createdAt: "asc" }] })
    .catch(() => []);

  return (
    <SiteShell slideshowCount={2}>
      <div className="pt-12 pb-24 sm:pt-16 sm:pb-32">
        <section className="relative mx-auto max-w-7xl px-5 sm:px-8">
          <Frost strength={80} className="p-8 sm:p-12 lg:p-16">
            <SectionMark number="VI" label="Where we eat" />

      <div className="mt-10 grid gap-10 lg:grid-cols-[1.45fr_1fr] lg:items-end lg:gap-20">
        <h1 className="font-display text-4xl font-light leading-[1.05] tracking-tightish text-ink sm:text-5xl lg:text-7xl">
          The
          <br />
          <span className="italic text-terracotta">restaurant</span> book.
        </h1>
        <p className="text-base leading-relaxed text-ink/75 sm:text-lg lg:pb-3">
          Our favourite places to eat around the villa. Book a couple of days
          ahead in summer — the good ones fill up.
        </p>
      </div>

      {restaurants.length === 0 ? (
        <div className="mt-14 rounded-3xl bg-white p-8 text-center text-ink/60 shadow-soft ring-1 ring-ink/5">
          We&apos;re still building the list — check back soon.
        </div>
      ) : (
        <ul className="mt-14 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {restaurants.map((r) => (
            <li
              key={r.id}
              className="flex flex-col rounded-3xl bg-white p-6 shadow-soft ring-1 ring-ink/5 sm:p-7"
            >
              <div>
                {r.area && (
                  <p className="text-xs uppercase tracking-wider text-ink/55">
                    {r.area}
                  </p>
                )}
                <h2 className="mt-1 font-display text-xl font-semibold text-ink">
                  {r.name}
                </h2>
              </div>
              {r.blurb && (
                <p className="mt-3 flex-1 text-sm leading-relaxed text-ink/75">
                  {r.blurb}
                </p>
              )}
              <div className="mt-4 flex flex-wrap gap-2">
                {r.website && (
                  <a
                    href={normalizeUrl(r.website)}
                    target="_blank"
                    rel="noreferrer noopener"
                    className="inline-flex items-center gap-1.5 rounded-full bg-sea/10 px-3 py-1.5 text-sm font-medium text-ocean hover:bg-sea/20"
                  >
                    <span aria-hidden>↗</span>
                    {siteHost(r.website)}
                  </a>
                )}
                {r.phone && (
                  <a
                    href={`tel:${r.phone.replace(/\s/g, "")}`}
                    className="inline-flex items-center gap-1.5 rounded-full bg-sand/60 px-3 py-1.5 text-sm font-medium text-ink hover:bg-sand"
                  >
                    <span aria-hidden>☎</span>
                    {r.phone}
                  </a>
                )}
              </div>
            </li>
          ))}
        </ul>
      )}

      <p className="mt-10 text-sm text-ink/55">
        Found a great place we should add? Mention it on{" "}
        <Link href="/my-bookings" className="text-ocean hover:underline">
          your booking thread
        </Link>
        .
      </p>
          </Frost>
        </section>
      </div>
    </SiteShell>
  );
}
