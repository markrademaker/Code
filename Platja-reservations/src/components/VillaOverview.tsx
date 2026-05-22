import { SectionMark, SunMark } from "@/components/Marks";

const features = [
  { title: "Private pool", description: "Sun terrace, sea views, day beds" },
  { title: "Four bedrooms", description: "Sleeps eight comfortably" },
  { title: "Open kitchen", description: "Fully equipped, BBQ, alfresco" },
  { title: "Mas Nou", description: "Quiet hillside neighbourhood" },
  { title: "Golf next door", description: "On the Mas Nou course" },
  { title: "Fast Wi-Fi", description: "Work from the terrace" },
];

export function VillaOverview() {
  return (
    <section
      id="overview"
      className="relative mx-auto max-w-7xl px-5 pt-16 sm:px-8 sm:pt-24 lg:pt-32"
    >
      <SectionMark number="I" label="The villa" />

      <div className="mt-10 grid gap-12 lg:grid-cols-[1.45fr_1fr] lg:items-end lg:gap-20">
        <div>
          <h1 className="font-display text-5xl font-semibold leading-[1.02] tracking-tight text-ink sm:text-7xl lg:text-[5.5rem]">
            White walls,
            <br />
            <span className="italic text-terracotta">pine</span>, and the
            <br />
            bay <span className="italic">below</span>.
          </h1>
        </div>

        <div className="lg:pb-3">
          <p className="text-base leading-relaxed text-ink/75 sm:text-lg">
            Our family villa sits in <strong className="font-medium text-ink">Mas Nou</strong>,
            a calm residential hillside above Platja d&apos;Aro. From the
            terrace you look out over umbrella pines and the Mediterranean —
            a fifteen-minute amble down to the coves of the Costa Brava.
          </p>
          <p className="mt-4 text-base leading-relaxed text-ink/75 sm:text-lg">
            We let it to friends, family, and a small number of guests each
            year. Bring your people, pick your dates, we&apos;ll reply by
            email.
          </p>
          <div className="mt-6 flex items-center gap-3 text-terracotta">
            <span className="h-px w-12 bg-terracotta/40" />
            <SunMark className="h-3.5 w-3.5" />
            <span className="font-display text-sm italic">
              Platja d&apos;Aro, Costa Brava
            </span>
          </div>
        </div>
      </div>

      <ol className="mt-16 grid grid-cols-1 overflow-hidden rounded-3xl bg-white/60 shadow-soft ring-1 ring-ink/5 sm:grid-cols-2 lg:grid-cols-3">
        {features.map((f, i) => (
          <li
            key={f.title}
            className={`relative flex items-start gap-4 p-6 sm:p-7 ${
              i % 2 ? "sm:border-l sm:border-ink/5" : ""
            } ${i % 3 !== 0 ? "lg:border-l lg:border-ink/5" : ""} ${
              i % 2 ? "sm:[&:nth-child(odd)]:border-l-0" : ""
            } ${i < features.length - (features.length % 2 || 2) ? "border-b border-ink/5" : ""}`}
          >
            <span className="font-display text-xl font-semibold italic text-terracotta">
              {String(i + 1).padStart(2, "0")}
            </span>
            <div>
              <p className="font-display text-lg font-semibold text-ink">
                {f.title}
              </p>
              <p className="mt-1 text-sm text-ink/65">{f.description}</p>
            </div>
          </li>
        ))}
      </ol>
    </section>
  );
}
