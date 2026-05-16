const SEASONS = [
  {
    name: "Low season",
    months: "Nov – Mar",
    weekly: "€1.400",
    note: "Quiet beaches and crisp light.",
    accent: "bg-sea/10 text-ocean ring-sea/20",
  },
  {
    name: "Mid season",
    months: "Apr – Jun · Sep – Oct",
    weekly: "€2.100",
    note: "Long sunny days, perfect for hiking and the pool.",
    accent: "bg-sunset/15 text-terracotta ring-sunset/30",
    featured: true,
  },
  {
    name: "High season",
    months: "Jul – Aug",
    weekly: "€2.800",
    note: "Warm sea, peak summer, lively coves.",
    accent: "bg-terracotta/10 text-terracotta ring-terracotta/20",
  },
];

export function Rates() {
  return (
    <section className="mx-auto max-w-6xl px-5 py-16 sm:px-6 sm:py-20">
      <div className="max-w-2xl">
        <p className="text-xs uppercase tracking-[0.2em] text-ink/55">
          Rates
        </p>
        <h2 className="mt-2 font-display text-3xl font-semibold sm:text-4xl">
          Indicative weekly rates
        </h2>
        <p className="mt-3 text-base text-ink/70 sm:text-lg">
          Stays of one week or longer. Get in touch for shorter breaks and
          last-minute openings.
        </p>
      </div>

      <div className="mt-10 grid gap-4 md:grid-cols-3">
        {SEASONS.map((s) => (
          <div
            key={s.name}
            className={`flex flex-col rounded-3xl bg-white p-6 shadow-soft ring-1 ring-ink/5 ${
              s.featured ? "md:scale-[1.02]" : ""
            }`}
          >
            <div className="flex items-center justify-between">
              <span
                className={`rounded-full px-2.5 py-0.5 text-[11px] font-medium uppercase tracking-wider ring-1 ${s.accent}`}
              >
                {s.name}
              </span>
              {s.featured && (
                <span className="text-[11px] font-medium uppercase tracking-wider text-terracotta">
                  Most loved
                </span>
              )}
            </div>
            <p className="mt-5 font-display text-3xl font-semibold text-ink">
              {s.weekly}
              <span className="ml-1 text-base font-normal text-ink/55">/ week</span>
            </p>
            <p className="mt-1 text-sm text-ink/60">{s.months}</p>
            <p className="mt-4 text-sm leading-relaxed text-ink/70">{s.note}</p>
          </div>
        ))}
      </div>
      <p className="mt-6 text-xs text-ink/50">
        Rates include cleaning at end of stay. Final price confirmed in your
        booking response.
      </p>
    </section>
  );
}
