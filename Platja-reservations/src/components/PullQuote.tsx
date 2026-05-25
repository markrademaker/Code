export function PullQuote() {
  return (
    <section className="mx-auto max-w-5xl px-5 py-32 text-center sm:px-8 sm:py-44">
      <span className="font-display text-3xl italic text-sunset">&ldquo;</span>
      <p
        className="mx-auto mt-2 max-w-3xl font-display text-3xl font-light italic leading-snug text-whitewash sm:text-5xl"
        style={{ textShadow: "0 2px 22px rgba(0,0,0,0.45)" }}
      >
        From the terrace you look out over umbrella pines and the
        Mediterranean — a fifteen-minute amble down to the coves.
      </p>
      <p className="mt-7 text-[11px] uppercase tracking-[0.35em] text-whitewash/80">
        — the owners
      </p>
    </section>
  );
}
