import type { Metadata } from "next";
import { format, parseISO } from "date-fns";
import { fetchForecast, describeWeatherCode } from "@/lib/weather";
import { PLATJA_CLIMATE } from "@/lib/climate";
import { SectionMark } from "@/components/Marks";

export const metadata: Metadata = {
  title: "Weather in Platja d'Aro",
  description:
    "Live seven-day forecast for Platja d'Aro on the Costa Brava plus typical monthly climate so you can plan when to visit Villa Mas Nou.",
  alternates: { canonical: "/weather" },
};

export const revalidate = 1800;

export default async function WeatherPage() {
  const forecast = await fetchForecast();
  const currentMonthIndex = new Date().getMonth();

  return (
    <main className="mx-auto max-w-7xl px-5 pt-16 pb-20 sm:px-8 sm:pt-24 sm:pb-28">
      <SectionMark number="V" label="Costa Brava forecast" />

      <div className="mt-10 grid gap-10 lg:grid-cols-[1.45fr_1fr] lg:items-end lg:gap-20">
        <h1 className="font-display text-4xl font-light leading-[1.05] tracking-tightish text-ink sm:text-5xl lg:text-7xl">
          Sun, <span className="italic text-terracotta">sea</span>,
          <br />
          and the in-between.
        </h1>
        <p className="text-base leading-relaxed text-ink/75 sm:text-lg lg:pb-3">
          A live seven-day forecast for the villa, plus the typical climate
          for each month so you can plan your stay.
        </p>
      </div>

      <section className="mt-16">
        <h2 className="font-display text-2xl font-semibold sm:text-3xl">
          Next 7 days
        </h2>
        {forecast ? (
          <>
            <div className="mt-4 rounded-2xl bg-gradient-to-br from-sea to-deep p-6 text-white shadow-sm">
              <p className="text-sm uppercase tracking-wider text-white/80">
                Right now
              </p>
              <div className="mt-2 flex items-center gap-4">
                <span className="text-5xl">
                  {describeWeatherCode(forecast.current.weatherCode).icon}
                </span>
                <div>
                  <p className="text-4xl font-semibold">
                    {Math.round(forecast.current.temperatureC)}°C
                  </p>
                  <p className="text-white/80">
                    {describeWeatherCode(forecast.current.weatherCode).label}
                  </p>
                </div>
              </div>
            </div>

            <ul className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              {forecast.daily.map((d) => {
                const { label, icon } = describeWeatherCode(d.weatherCode);
                return (
                  <li
                    key={d.date}
                    className="rounded-2xl bg-white p-4 shadow-sm ring-1 ring-deep/5"
                  >
                    <p className="text-sm font-medium text-deep/70">
                      {format(parseISO(d.date), "EEE d MMM")}
                    </p>
                    <div className="mt-2 flex items-center justify-between">
                      <span className="text-3xl">{icon}</span>
                      <p className="text-right">
                        <span className="text-lg font-semibold">
                          {Math.round(d.highC)}°
                        </span>
                        <span className="ml-1 text-deep/50">
                          / {Math.round(d.lowC)}°
                        </span>
                      </p>
                    </div>
                    <p className="mt-2 text-xs text-deep/60">
                      {label} · {d.precipitationMm.toFixed(1)} mm rain
                    </p>
                  </li>
                );
              })}
            </ul>
            <p className="mt-3 text-xs text-deep/50">
              Forecast from Open-Meteo, updated every 30 minutes.
            </p>
          </>
        ) : (
          <p className="mt-4 rounded-2xl bg-white p-6 text-deep/70 shadow-sm ring-1 ring-deep/5">
            Forecast unavailable right now. Try again in a bit.
          </p>
        )}
      </section>

      <section className="mt-20">
        <h2 className="font-display text-2xl font-semibold sm:text-3xl">
          Typical weather by month
        </h2>
        <p className="mt-2 max-w-2xl text-sm text-deep/70 sm:text-base">
          Long-term averages for the Costa Brava — useful for picking when to
          visit.
        </p>
        <ul className="mt-6 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {PLATJA_CLIMATE.map((m, i) => {
            const isCurrent = i === currentMonthIndex;
            return (
              <li
                key={m.month}
                className={`rounded-2xl p-5 shadow-sm ring-1 ${
                  isCurrent
                    ? "bg-sea/15 ring-sea/30"
                    : "bg-white ring-deep/5"
                }`}
              >
                <div className="flex items-baseline justify-between">
                  <p className="font-display text-lg font-semibold">{m.month}</p>
                  {isCurrent && (
                    <span className="rounded-full bg-sea px-2 py-0.5 text-xs font-medium text-white">
                      Now
                    </span>
                  )}
                </div>
                <div className="mt-2 flex gap-4 text-sm text-deep/80">
                  <span>
                    <strong className="text-deep">{m.highC}°</strong> / {m.lowC}°
                  </span>
                  <span>Sea {m.seaC}°</span>
                  <span>{m.rainMm} mm</span>
                </div>
                <p className="mt-2 text-sm text-deep/70">{m.blurb}</p>
              </li>
            );
          })}
        </ul>
      </section>
    </main>
  );
}
