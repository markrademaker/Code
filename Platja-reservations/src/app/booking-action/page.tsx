import Link from "next/link";

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
    <main className="mx-auto max-w-2xl px-5 py-20 sm:px-6">
      <h1 className="font-display text-3xl font-semibold sm:text-4xl">{m.title}</h1>
      <p className="mt-4 text-lg text-deep/80">{m.body(searchParams.guest, searchParams.status)}</p>
      <Link
        href="/admin"
        className="mt-8 inline-block rounded-full bg-deep px-6 py-3 font-medium text-white"
      >
        Open admin dashboard
      </Link>
    </main>
  );
}
