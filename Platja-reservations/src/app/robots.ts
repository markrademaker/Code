import type { MetadataRoute } from "next";

function baseUrl(): string {
  return (
    process.env.APP_BASE_URL?.replace(/\/$/, "") ??
    process.env.NEXT_PUBLIC_APP_BASE_URL?.replace(/\/$/, "") ??
    "https://platja-reservations.vercel.app"
  );
}

export default function robots(): MetadataRoute.Robots {
  return {
    rules: [
      {
        userAgent: "*",
        allow: "/",
        disallow: [
          "/admin",
          "/admin/",
          "/api/",
          "/login",
          "/my-bookings",
          "/pay",
          "/weather",
          "/restaurants",
          "/house-rules",
          "/booking-action",
        ],
      },
    ],
    sitemap: `${baseUrl()}/sitemap.xml`,
    host: baseUrl(),
  };
}
