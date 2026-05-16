const features = [
  { title: "Private pool", description: "Sun terrace, sea views, day beds" },
  { title: "4 bedrooms", description: "Sleeps 8 comfortably" },
  { title: "Open kitchen", description: "Fully equipped, BBQ, alfresco dining" },
  { title: "Mas Nou", description: "Quiet hillside neighbourhood" },
  { title: "Golf next door", description: "On the Mas Nou golf course" },
  { title: "Fast Wi-Fi", description: "Work from the terrace if you must" },
];

export function VillaOverview() {
  return (
    <section
      id="overview"
      className="mx-auto max-w-6xl px-5 pt-24 sm:px-6 sm:pt-32 sm:pb-16 pb-12"
    >
      <div className="grid gap-12 md:grid-cols-2 md:items-start md:gap-16">
        <div>
          <p className="text-xs uppercase tracking-[0.2em] text-ink/55">
            The villa
          </p>
          <h2 className="mt-2 font-display text-3xl font-semibold leading-tight sm:text-4xl lg:text-5xl">
            White walls, pine, and the bay below
          </h2>
          <p className="mt-5 text-base leading-relaxed text-ink/75 sm:text-lg">
            Our family villa sits in Mas Nou, a calm residential neighbourhood
            on the hillside above Platja d&apos;Aro. From the terrace you look
            out over umbrella pines and the Mediterranean. It&apos;s a great
            base for exploring the Costa Brava — from coves and beaches to the
            medieval towns of the Empordà.
          </p>
          <p className="mt-4 text-base leading-relaxed text-ink/75 sm:text-lg">
            We rent it to friends, family and a small number of guests each
            year. Reach out with your dates and we&apos;ll confirm by email.
          </p>
        </div>
        <ul className="grid grid-cols-2 gap-3">
          {features.map((f, i) => (
            <li
              key={f.title}
              className={`rounded-2xl bg-white p-5 shadow-soft ring-1 ring-ink/5 ${
                i % 3 === 0 ? "" : ""
              }`}
            >
              <p className="font-semibold text-ink">{f.title}</p>
              <p className="mt-1 text-sm text-ink/65">{f.description}</p>
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}
