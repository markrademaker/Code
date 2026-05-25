import Link from "next/link";

export function Hero() {
  return (
    <section id="top" className="relative">
      <div className="mx-auto max-w-7xl px-5 pt-24 pb-40 sm:px-8 sm:pt-32 sm:pb-56 lg:pt-40 lg:pb-72">
        <div className="max-w-3xl">
          <div className="flex items-center gap-3 text-[11px] uppercase tracking-[0.32em] text-whitewash/85">
            <span className="h-px w-10 bg-whitewash/55" aria-hidden />
            <span>Platja d&apos;Aro · Costa Brava</span>
          </div>
          <h1
            className="mt-7 font-display font-light leading-[0.95] tracking-tightish text-whitewash"
            style={{
              fontSize: "clamp(3.25rem, 9.5vw, 9rem)",
              textShadow:
                "0 2px 30px rgba(0,0,0,0.35), 0 1px 2px rgba(0,0,0,0.2)",
            }}
          >
            White walls,
            <br />
            <span className="italic text-sunset">pine</span>, and the
            <br />
            bay <span className="italic">below</span>.
          </h1>
          <p
            className="mt-8 max-w-xl text-base leading-relaxed text-whitewash/90 sm:text-lg"
            style={{ textShadow: "0 1px 18px rgba(0,0,0,0.45)" }}
          >
            A whitewashed family villa above Platja d&apos;Aro. Private pool,
            sea views, the cove a fifteen-minute amble down. Bring your people,
            pick your dates — we&apos;ll reply by email.
          </p>
          <div className="mt-10 flex flex-wrap items-center gap-4">
            <Link
              href="#book"
              className="rounded-full bg-ocean px-7 py-3.5 text-xs font-medium uppercase tracking-wider text-whitewash shadow-glow hover:bg-ocean/90"
            >
              Request dates
            </Link>
            <Link
              href="/photos"
              className="rounded-full border border-whitewash/50 px-7 py-3.5 text-xs font-medium uppercase tracking-wider text-whitewash hover:bg-whitewash/10"
            >
              Tour the villa
            </Link>
          </div>
        </div>
      </div>

      <div className="pointer-events-none absolute inset-x-0 bottom-8 mx-auto flex w-fit flex-col items-center gap-2 text-whitewash/75">
        <span className="font-display text-sm italic">scroll</span>
        <span className="block h-9 w-px animate-pulse bg-whitewash/70" />
      </div>
    </section>
  );
}
