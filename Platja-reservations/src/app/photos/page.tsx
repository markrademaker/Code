import type { Metadata } from "next";
import { SectionMark } from "@/components/Marks";

export const metadata: Metadata = {
  title: "Photos — Villa Mas Nou",
  description: "Photos of the villa, the views and the surroundings.",
};

export default function PhotosPage() {
  return (
    <main className="mx-auto max-w-7xl px-5 pt-16 pb-20 sm:px-8 sm:pt-24 sm:pb-28">
      <SectionMark number="IV" label="The villa, in pictures" />

      <div className="mt-10 grid gap-10 lg:grid-cols-[1.45fr_1fr] lg:items-end lg:gap-20">
        <h1 className="font-display text-5xl font-semibold leading-[1.02] tracking-tight text-ink sm:text-7xl lg:text-[5.5rem]">
          A <span className="italic text-terracotta">moment</span>,
          <br />
          coming soon.
        </h1>
        <p className="text-base leading-relaxed text-ink/75 sm:text-lg lg:pb-3">
          Photos of the pool, the terrace, the bedrooms and the views are on
          their way. Drop image files into{" "}
          <code className="rounded bg-white px-1.5 py-0.5 font-mono text-sm">
            public/photos/
          </code>{" "}
          and we&apos;ll wire up the gallery.
        </p>
      </div>

      <div className="mt-14 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {Array.from({ length: 6 }).map((_, i) => (
          <div
            key={i}
            className="relative aspect-[4/3] overflow-hidden rounded-3xl bg-gradient-to-br from-sand via-stone/30 to-sea/15 ring-1 ring-ink/10"
            aria-hidden
          >
            <span className="absolute left-5 top-4 font-display text-xl italic text-terracotta/70">
              {String(i + 1).padStart(2, "0")}
            </span>
          </div>
        ))}
      </div>
    </main>
  );
}
