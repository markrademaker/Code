import type { MetadataRoute } from "next";

function baseUrl(): string {
  return (
    process.env.APP_BASE_URL?.replace(/\/$/, "") ??
    process.env.NEXT_PUBLIC_APP_BASE_URL?.replace(/\/$/, "") ??
    "https://platja-reservations.vercel.app"
  );
}

export default function sitemap(): MetadataRoute.Sitemap {
  const base = baseUrl();
  const now = new Date();
  return [
    { url: `${base}/`, lastModified: now, changeFrequency: "weekly", priority: 1 },
    { url: `${base}/photos`, lastModified: now, changeFrequency: "monthly", priority: 0.7 },
  ];
}
