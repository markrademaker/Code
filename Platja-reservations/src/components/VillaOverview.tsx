import { SectionMark, SunMark } from "@/components/Marks";
import { Frost } from "@/components/Frost";

const features = [
  { title: "Private pool", description: "Sun terrace, sea views, day beds" },
  { title: "Four bedrooms", description: "Sleeps eight comfortably" },
  { title: "Open kitchen", description: "Fully equipped, BBQ, alfresco" },
  { title: "Mas Nou", description: "Quiet hillside neighbourhood" },
  { title: "Golf next door", description: "On the Mas Nou course" },
  { title: "Fast Wi-Fi", description: "Work from the terrace" },
];

export function VillaOverview({ frostStrength = 80 }: { frostStrength?: number }) {
  return (
    <section id="villa" className="relative mx-auto max-w-7xl px-5 sm:px-8">
      <Frost strength={frostStrength} className="p-8 sm:p-12 lg:p-16">
        <SectionMark number="I" label="The villa" />

        <div className="mt-10 grid gap-12 lg:grid-cols-[1.45fr_1fr] lg:items-end lg:gap-20">
          <h2 className="font-display text-4xl font-light leading-[1.05] tracking-tightish text-ink sm:text-5xl lg:text-7xl">
            White walls,
            <br />
            <span className="text-terracotta">pine</span>, and the
            <br />
            bay <span className="italic">below</span>.
          </h2>

          <div className="lg:pb-3">
            <p className="text-sm leading-relaxed text-ink/75 sm:text-base">
              Our family villa sits in{" "}
              <strong className="font-medium text-ink">Mas Nou</strong>, a calm
              residential hillside above Platja d&apos;Aro. From the terrace
              you look out over umbrella pines and the Mediterranean — a
              fifteen-minute amble down to the coves of the Costa Brava.
            </p>
            <p className="mt-3 text-sm leading-relaxed text-ink/75 sm:text-base">
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

        <ol className="mt-14 grid grid-cols-1 overflow-hidden rounded-3xl bg-white/55 ring-1 ring-ink/5 sm:grid-cols-2 lg:grid-cols-3">
          {features.map((f, i) => (
            <li
              key={f.title}
              className={`relative flex items-start gap-4 p-6 sm:p-7 ${
                i % 3 !== 0 ? "lg:border-l lg:border-ink/5" : ""
              } ${
                i % 2 !== 0
                  ? "sm:border-l sm:border-ink/5 lg:border-l lg:border-ink/5"
                  : ""
              } ${
                i < features.length - 3 ? "lg:border-b lg:border-ink/5" : ""
              } ${i < features.length - 2 ? "sm:border-b sm:border-ink/5" : ""}`}
            >
              <span className="font-display text-xl font-light italic text-terracotta">
                {String(i + 1).padStart(2, "0")}
              </span>
              <div>
                <p className="font-display text-lg text-ink">{f.title}</p>
                <p className="mt-1 text-sm text-ink/65">{f.description}</p>
              </div>
            </li>
          ))}
        </ol>
      </Frost>
    </section>
  );
}
