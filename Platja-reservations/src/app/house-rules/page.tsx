import type { Metadata } from "next";
import { SectionMark } from "@/components/Marks";

export const metadata: Metadata = {
  title: "House rules",
  description:
    "Everything to know before your stay at Villa Mas Nou — arrival, departure, pool, quiet hours, pets, and the security deposit.",
  alternates: { canonical: "/house-rules" },
};

const RULES: { title: string; items: string[] }[] = [
  {
    title: "Arrival & departure",
    items: [
      "Check-in from 16:00, check-out by 11:00.",
      "Let us know your rough arrival time the day before so we can be there.",
      "Self-check-in is fine if you tell us in advance — we'll send the gate code.",
    ],
  },
  {
    title: "Guests",
    items: [
      "Sleeps 8 across 4 bedrooms — no extra guests without asking first.",
      "Day visitors are welcome; just give us a heads up.",
      "No events or parties without prior agreement.",
    ],
  },
  {
    title: "Pool & garden",
    items: [
      "The pool is unfenced — please supervise children at all times.",
      "Last swim by 22:00 out of respect for the neighbours.",
      "No glass around the pool deck.",
    ],
  },
  {
    title: "Quiet hours",
    items: [
      "22:00 – 08:00. Mas Nou is a residential area and sound travels.",
      "Music outside should stay at conversation level after 22:00.",
    ],
  },
  {
    title: "Smoking & pets",
    items: [
      "Strictly no smoking inside the villa.",
      "Smoking allowed on the terrace; please use the ashtrays.",
      "Well-behaved dogs welcome with prior arrangement — there's an extra cleaning fee.",
    ],
  },
  {
    title: "Cleaning, damages & deposit",
    items: [
      "A cleaning fee is included in your booking.",
      "We ask for a refundable security deposit on arrival, returned within a week of check-out.",
      "Accidents happen — just tell us so we can sort it out.",
    ],
  },
];

export default function HouseRulesPage() {
  return (
    <main className="mx-auto max-w-7xl px-5 pt-16 pb-20 sm:px-8 sm:pt-24 sm:pb-28">
      <SectionMark number="VII" label="Before you arrive" />

      <div className="mt-10 grid gap-10 lg:grid-cols-[1.45fr_1fr] lg:items-end lg:gap-20">
        <h1 className="font-display text-4xl font-light leading-[1.05] tracking-tightish text-ink sm:text-5xl lg:text-7xl">
          House
          <br />
          <span className="italic text-terracotta">rules</span>.
        </h1>
        <p className="text-base leading-relaxed text-ink/75 sm:text-lg lg:pb-3">
          Nothing scary — just the things that keep the villa and the
          neighbourhood happy.
        </p>
      </div>

      <div className="mt-14 grid gap-4 lg:max-w-3xl">
        {RULES.map((s) => (
          <section
            key={s.title}
            className="rounded-3xl bg-white p-6 shadow-soft ring-1 ring-ink/5 sm:p-7"
          >
            <h2 className="font-display text-xl font-semibold text-ink">
              {s.title}
            </h2>
            <ul className="mt-3 grid gap-2">
              {s.items.map((it) => (
                <li
                  key={it}
                  className="flex gap-3 text-sm leading-relaxed text-ink/75 sm:text-base"
                >
                  <span aria-hidden className="mt-2 inline-block h-1.5 w-1.5 shrink-0 rounded-full bg-sea" />
                  <span>{it}</span>
                </li>
              ))}
            </ul>
          </section>
        ))}
      </div>
    </main>
  );
}
