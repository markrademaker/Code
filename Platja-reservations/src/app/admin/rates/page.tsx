import { prisma } from "@/lib/db";
import { RateAdmin } from "@/components/RateAdmin";
import { requireAdmin } from "@/lib/require-admin";

export const dynamic = "force-dynamic";

export default async function AdminRatesPage() {
  await requireAdmin("/admin/rates");
  const rates = await prisma.ratePeriod
    .findMany({ orderBy: { startDate: "asc" } })
    .catch(() => []);
  return (
    <RateAdmin
      initial={rates.map((r) => ({
        id: r.id,
        startDate: r.startDate.toISOString().slice(0, 10),
        endDate: r.endDate.toISOString().slice(0, 10),
        nightlyRateCents: r.nightlyRateCents,
        label: r.label,
      }))}
    />
  );
}
