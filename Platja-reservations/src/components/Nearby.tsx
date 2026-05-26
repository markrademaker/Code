import { SectionMark } from "@/components/Marks";
import { Frost } from "@/components/Frost";

type Place = {
  d: string;
  title: string;
  note: string;
};

const PLACES: Place[] = [
  { d: "2 min", title: "Mas Nou golf", note: "On the course, walk to the first tee" },
  { d: "5 min", title: "Cala Rovira", note: "Pine-cliffed cove, calm water" },
  { d: "8 min", title: "Platja d'Aro", note: "Town beach, paseo, restaurants" },
  { d: "20 min", title: "S'Agaró", note: "Cliff path, modernista villas" },
  { d: "30 min", title: "Begur · Pals", note: "Stone villages, the inland Empordà" },
  { d: "45 min", title: "Girona old town", note: "Cathedral, river bridges, dinner" },
];

export function Nearby({ frostStrength = 80 }: { frostStrength?: number }) {
  return (
    <section id="nearby" className="relative mx-auto max-w-7xl px-5 sm:px-8">
      <Frost strength={frostStrength} className="p-8 sm:p-12 lg:p-16">
        <SectionMark number="IV" label="Nearby" />
        <h2 className="mt-8 font-display text-4xl font-light leading-[1.05] tracking-tightish sm:text-5xl">
          A few minutes{" "}
          <span className="text-terracotta">from the door</span>.
        </h2>
        <div className="mt-10 grid gap-x-10 gap-y-6 sm:grid-cols-2 lg:grid-cols-3">
          {PLACES.map((p) => (
            <div
              key={p.title}
              className="flex items-baseline gap-5 border-b border-ink/10 pb-5"
            >
              <span className="whitespace-nowrap font-display text-2xl italic text-terracotta">
                {p.d}
              </span>
              <div>
                <p className="font-display text-lg text-ink">{p.title}</p>
                <p className="mt-0.5 text-sm text-ink/65">{p.note}</p>
              </div>
            </div>
          ))}
        </div>
      </Frost>
    </section>
  );
}
