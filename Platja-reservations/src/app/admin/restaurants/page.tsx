import { prisma } from "@/lib/db";
import { RestaurantAdmin } from "@/components/RestaurantAdmin";

export const dynamic = "force-dynamic";

export default async function AdminRestaurantsPage() {
  const restaurants = await prisma.restaurant
    .findMany({ orderBy: [{ sortOrder: "asc" }, { createdAt: "asc" }] })
    .catch(() => []);
  return (
    <RestaurantAdmin
      initial={restaurants.map((r) => ({
        id: r.id,
        name: r.name,
        website: r.website,
        phone: r.phone,
        area: r.area,
        blurb: r.blurb,
        sortOrder: r.sortOrder,
      }))}
    />
  );
}
