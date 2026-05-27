import Link from "next/link";
import { SectionMark } from "@/components/Marks";

const MESSAGES: Record<string, { title: string; body: (g?: string, s?: string) => string }> = {
  ok: {
    title: "Done",
    body: (g, s) => `${g ?? "The booking"} has been marked as ${s?.toLowerCase() ?? "updated"} and the guest has been emailed.`,
  },
  already: {
    title: "Already actioned",
    body: (g, s) => `${g ?? "This booking"} is already ${s?.toLowerCase() ?? "set"}. Open the admin dashboard to make changes.`,
  },
  conflict: {
    title: "Dates no longer available",
    body: (g) => `Another booking now overlaps with ${g ?? "this request"}. Open the admin dashboard to resolve.`,
  },
  notfound: {
    title: "Booking not found",
    body: () => "This booking may have been removed.",
  },
  invalid: {
    title: "Link expired or invalid",
    body: () => "This action link is no longer valid. Open the admin dashboard instead.",
  },
};

export default function BookingActionPage({
  searchParams,
}: {
  searchParams: { state?: string; guest?: string; status?: string };
}) {
  const key = searchParams.state ?? "invalid";
  const m = MESSAGES[key] ?? MESSAGES.invalid;
  return (
    <main className="mx-auto max-w-7xl px-5 pt-16 pb-20 sm:px-8 sm:pt-24 sm:pb-28">
      <SectionMark number="VIII" label="Email action" />
      <div className="mt-10 grid gap-10 lg:grid-cols-[1.45fr_1fr] lg:items-end lg:gap-20">
        <h1 className="font-display text-4xl font-light leading-[1.05] tracking-tightish text-ink sm:text-5xl lg:text-7xl">
          {m.title}
          <span>.</span>
        </h1>
        <p className="text-base leading-relaxed text-ink/75 sm:text-lg lg:pb-3">
          {m.body(searchParams.guest, searchParams.status)}
        </p>
      </div>
      <Link
        href="/admin"
        className="mt-12 inline-block rounded-full bg-ocean px-7 py-3.5 font-medium text-whitewash shadow-glow hover:bg-ocean/90"
      >
        Open admin dashboard
      </Link>
    </main>
  );
}
