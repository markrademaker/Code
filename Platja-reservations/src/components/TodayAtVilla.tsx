import { describeWeatherCode, fetchToday } from "@/lib/weather";

export async function TodayAtVilla() {
  const today = await fetchToday();
  if (!today) {
    return (
      <section className="bg-whitewash py-10">
        <div className="mx-auto max-w-6xl px-5 sm:px-6">
          <div className="rounded-3xl bg-white p-6 text-sm text-ink/60 shadow-soft ring-1 ring-ink/5">
            Today&apos;s weather isn&apos;t available right now.
          </div>
        </div>
      </section>
    );
  }

  const { label, icon } = describeWeatherCode(today.weatherCode);
  const tiles = [
    { label: "Now", value: `${Math.round(today.temperatureC)}°`, sub: label, icon },
    {
      label: "High / Low",
      value: `${Math.round(today.highC)}° / ${Math.round(today.lowC)}°`,
      sub: "Today",
    },
    {
      label: "Rain",
      value: `${today.precipitationMm.toFixed(1)} mm`,
      sub: today.precipitationMm > 0.5 ? "Likely showers" : "Dry day",
    },
    today.seaTempC != null
      ? {
          label: "Sea",
          value: `${Math.round(today.seaTempC)}°`,
          sub: "Surface",
          icon: "🌊",
        }
      : null,
    { label: "Sunrise", value: today.sunrise, sub: "Costa Brava", icon: "🌅" },
    { label: "Sunset", value: today.sunset, sub: "Costa Brava", icon: "🌇" },
  ].filter(Boolean) as { label: string; value: string; sub: string; icon?: string }[];

  return (
    <section className="bg-whitewash py-10 sm:py-14">
      <div className="mx-auto max-w-6xl px-5 sm:px-6">
        <div className="flex items-end justify-between">
          <div>
            <p className="text-xs uppercase tracking-[0.2em] text-ink/55">
              Right now
            </p>
            <h2 className="mt-1 font-display text-2xl font-semibold sm:text-3xl">
              Today at the villa
            </h2>
          </div>
          <a
            href="/weather"
            className="hidden text-sm font-medium text-ocean hover:underline sm:inline"
          >
            7-day forecast →
          </a>
        </div>

        <ul className="mt-6 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
          {tiles.map((t) => (
            <li
              key={t.label}
              className="rounded-2xl bg-white p-4 shadow-soft ring-1 ring-ink/5"
            >
              <div className="flex items-center justify-between">
                <p className="text-[11px] uppercase tracking-wider text-ink/55">
                  {t.label}
                </p>
                {t.icon && <span className="text-lg">{t.icon}</span>}
              </div>
              <p className="mt-1 font-display text-2xl font-semibold text-ink">
                {t.value}
              </p>
              <p className="text-xs text-ink/55">{t.sub}</p>
            </li>
          ))}
        </ul>
        <a
          href="/weather"
          className="mt-4 block text-center text-sm font-medium text-ocean hover:underline sm:hidden"
        >
          7-day forecast →
        </a>
      </div>
    </section>
  );
}
