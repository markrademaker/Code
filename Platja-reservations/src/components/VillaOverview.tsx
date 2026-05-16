const features = [
  { title: "4 bedrooms", description: "Sleeps up to 8 guests comfortably" },
  { title: "Private pool", description: "Sun terrace with sea views" },
  { title: "Mas Nou", description: "Quiet residential area in the hills" },
  { title: "5 min to beach", description: "Short drive to Platja d'Aro" },
  { title: "Golf nearby", description: "Adjacent to the Mas Nou golf club" },
  { title: "Fully equipped", description: "Kitchen, BBQ, Wi-Fi, parking" },
];

export function VillaOverview() {
  return (
    <section id="overview" className="mx-auto max-w-6xl px-5 py-14 sm:px-6 sm:py-20">
      <div className="grid gap-10 md:grid-cols-2 md:items-center md:gap-12">
        <div>
          <h2 className="font-display text-2xl font-semibold sm:text-3xl lg:text-4xl">
            The villa
          </h2>
          <p className="mt-4 text-base leading-relaxed text-deep/80 sm:text-lg">
            Our family villa sits in Mas Nou, a calm residential neighbourhood
            on the hillside above Platja d&apos;Aro. From the terrace you look
            out over pine forests and the bay. It&apos;s a great base for
            exploring the Costa Brava — from coves and beaches to the medieval
            towns of the Empordà.
          </p>
          <p className="mt-4 text-base leading-relaxed text-deep/80 sm:text-lg">
            We rent it out to friends, family and a small number of guests each
            year. Reach out with your dates and we&apos;ll confirm by email.
          </p>
        </div>
        <ul className="grid grid-cols-2 gap-3 sm:gap-4">
          {features.map((f) => (
            <li
              key={f.title}
              className="rounded-2xl bg-white/70 p-4 shadow-sm ring-1 ring-deep/5 sm:p-5"
            >
              <p className="font-semibold text-deep">{f.title}</p>
              <p className="mt-1 text-sm text-deep/70">{f.description}</p>
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}
