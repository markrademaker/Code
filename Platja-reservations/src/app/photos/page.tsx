import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Photos — Villa Mas Nou",
  description: "Photos of the villa, the views and the surroundings.",
};

export default function PhotosPage() {
  return (
    <main className="mx-auto max-w-6xl px-5 py-10 sm:px-6 sm:py-14">
      <p className="text-xs uppercase tracking-[0.25em] text-deep/60 sm:text-sm">
        Mas Nou · Platja d&apos;Aro
      </p>
      <h1 className="mt-2 font-display text-3xl font-semibold sm:text-4xl">
        Photos
      </h1>
      <p className="mt-3 max-w-2xl text-base text-deep/70 sm:text-lg">
        Photos of the villa, the pool, the views and the surroundings are
        coming soon.
      </p>

      <div className="mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {Array.from({ length: 6 }).map((_, i) => (
          <div
            key={i}
            className="aspect-[4/3] rounded-2xl bg-gradient-to-br from-sand to-sea/20 ring-1 ring-deep/10"
            aria-hidden
          />
        ))}
      </div>

      <p className="mt-8 text-sm text-deep/60">
        Drop image files into <code className="rounded bg-white px-1.5 py-0.5">public/photos/</code>{" "}
        and we&apos;ll wire up the gallery.
      </p>
    </main>
  );
}
