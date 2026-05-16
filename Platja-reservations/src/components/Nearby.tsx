type Place = {
  name: string;
  blurb: string;
  distance: string;
  tag: string;
};

const PLACES: Place[] = [
  {
    name: "Cala Sa Cova",
    blurb: "A quiet pine-fringed cove you can walk to from the villa.",
    distance: "10 min walk",
    tag: "Beach",
  },
  {
    name: "Platja d'Aro old town",
    blurb: "Tapas, ice cream, and the seafront promenade.",
    distance: "5 min drive",
    tag: "Town",
  },
  {
    name: "Mas Nou golf course",
    blurb: "18 holes literally over the wall — book a tee time at reception.",
    distance: "Next door",
    tag: "Golf",
  },
  {
    name: "S'Agaró",
    blurb: "Camí de Ronda coastal path past Modernista villas and hidden coves.",
    distance: "10 min drive",
    tag: "Walk",
  },
  {
    name: "Calonge medieval village",
    blurb: "Stone streets, a castle, and great Sunday lunches.",
    distance: "15 min drive",
    tag: "Day trip",
  },
  {
    name: "Begur & the Empordà",
    blurb: "Hilltop town, wine country, and some of the wildest coves.",
    distance: "40 min drive",
    tag: "Day trip",
  },
];

export function Nearby() {
  return (
    <section className="bg-whitewash py-16 sm:py-20">
      <div className="mx-auto max-w-6xl px-5 sm:px-6">
        <div className="max-w-2xl">
          <p className="text-xs uppercase tracking-[0.2em] text-ink/55">
            Around the villa
          </p>
          <h2 className="mt-2 font-display text-3xl font-semibold sm:text-4xl">
            Things to do nearby
          </h2>
          <p className="mt-3 text-base text-ink/70 sm:text-lg">
            A handful of our favourite places, in walking distance or a short
            drive away.
          </p>
        </div>

        <ul className="mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {PLACES.map((p) => (
            <li
              key={p.name}
              className="group rounded-3xl bg-white p-6 shadow-soft ring-1 ring-ink/5 transition hover:-translate-y-0.5 hover:shadow-glow"
            >
              <div className="flex items-center justify-between">
                <span className="rounded-full bg-sand/70 px-2.5 py-0.5 text-[11px] font-medium uppercase tracking-wider text-ink/70">
                  {p.tag}
                </span>
                <span className="text-xs text-ink/50">{p.distance}</span>
              </div>
              <h3 className="mt-4 font-display text-xl font-semibold text-ink">
                {p.name}
              </h3>
              <p className="mt-2 text-sm leading-relaxed text-ink/70">
                {p.blurb}
              </p>
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}
