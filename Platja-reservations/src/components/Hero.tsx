export function Hero() {
  return (
    <section className="relative overflow-hidden bg-sand-wash">
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 bg-sun-fade"
      />
      <div
        aria-hidden
        className="pointer-events-none absolute -top-24 -right-24 h-72 w-72 rounded-full bg-sunset/30 blur-3xl sm:h-96 sm:w-96"
      />
      <div
        aria-hidden
        className="pointer-events-none absolute -bottom-24 -left-24 h-72 w-72 rounded-full bg-sea/25 blur-3xl sm:h-96 sm:w-96"
      />

      <div className="relative mx-auto max-w-6xl px-5 py-20 sm:px-6 sm:py-28 lg:py-36">
        <p className="mb-4 inline-flex items-center gap-2 rounded-full bg-white/70 px-3 py-1 text-[11px] uppercase tracking-[0.2em] text-ink/70 backdrop-blur sm:text-xs">
          <span className="h-1.5 w-1.5 rounded-full bg-terracotta"></span>
          Mas Nou · Platja d&apos;Aro · Costa Brava
        </p>
        <h1 className="font-display text-4xl font-semibold leading-[1.05] text-ink sm:text-6xl lg:text-7xl">
          A quiet villa <span className="italic text-ocean">above</span>
          <br className="hidden sm:block" /> the Mediterranean
        </h1>
        <p className="mt-6 max-w-2xl text-base text-ink/70 sm:text-lg">
          Whitewashed walls, a private pool, pine and the bay below. Our
          family villa is a few minutes from the coves and old town of Platja
          d&apos;Aro — bring your people, pick your dates, and we&apos;ll
          confirm by email.
        </p>
        <div className="mt-10 flex flex-col gap-3 sm:flex-row sm:flex-wrap">
          <a
            href="#book"
            className="rounded-full bg-ocean px-7 py-3.5 text-center font-medium text-whitewash shadow-glow transition hover:bg-ocean/90"
          >
            Request a booking
          </a>
          <a
            href="#availability"
            className="rounded-full border border-ink/15 bg-white/70 px-7 py-3.5 text-center font-medium text-ink backdrop-blur transition hover:bg-white"
          >
            View availability
          </a>
        </div>
      </div>

      <QuickFacts />
    </section>
  );
}

function QuickFacts() {
  const facts = [
    { label: "Sleeps", value: "8" },
    { label: "Bedrooms", value: "4" },
    { label: "Pool", value: "Private" },
    { label: "Beach", value: "5 min" },
  ];
  return (
    <div className="relative mx-auto -mb-10 max-w-5xl translate-y-10 px-5 sm:-mb-12 sm:px-6">
      <div className="grid grid-cols-2 gap-px overflow-hidden rounded-3xl bg-ink/5 shadow-soft ring-1 ring-ink/5 sm:grid-cols-4">
        {facts.map((f) => (
          <div key={f.label} className="bg-white px-5 py-5 text-center">
            <p className="font-display text-2xl font-semibold text-ink">
              {f.value}
            </p>
            <p className="mt-1 text-xs uppercase tracking-wider text-ink/55">
              {f.label}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}
