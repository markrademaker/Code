export function Hero() {
  return (
    <section className="relative overflow-hidden bg-gradient-to-br from-deep via-sea to-terracotta text-white">
      <div className="mx-auto max-w-6xl px-5 py-16 sm:px-6 sm:py-24 lg:py-32">
        <p className="mb-3 text-xs uppercase tracking-[0.25em] text-white/80 sm:text-sm sm:tracking-[0.3em]">
          Mas Nou · Platja d&apos;Aro · Costa Brava
        </p>
        <h1 className="font-display text-3xl font-semibold leading-tight sm:text-5xl lg:text-6xl">
          A quiet villa above the Mediterranean
        </h1>
        <p className="mt-5 max-w-2xl text-base text-white/90 sm:mt-6 sm:text-lg">
          Spend your holiday in a private villa in the hills of Mas Nou, a few
          minutes from the beaches and old town of Platja d&apos;Aro. Check
          availability and request a booking — we&apos;ll get back to you
          personally.
        </p>
        <div className="mt-8 flex flex-col gap-3 sm:mt-10 sm:flex-row sm:flex-wrap sm:gap-4">
          <a
            href="#availability"
            className="rounded-full bg-white px-6 py-3 text-center font-medium text-deep shadow-lg transition hover:bg-sand"
          >
            View availability
          </a>
          <a
            href="#book"
            className="rounded-full border border-white/60 px-6 py-3 text-center font-medium text-white transition hover:bg-white/10"
          >
            Request a booking
          </a>
        </div>
      </div>
    </section>
  );
}
