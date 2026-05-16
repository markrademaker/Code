import type { Metadata } from "next";
import { format, parseISO } from "date-fns";
import { fetchForecast, describeWeatherCode } from "@/lib/weather";
import { PLATJA_CLIMATE } from "@/lib/climate";

export const metadata: Metadata = {
  title: "Weather — Villa Mas Nou",
  description:
    "Live 7-day forecast for Platja d'Aro plus typical climate by month, to help you plan your stay.",
};

export const revalidate = 1800;

export default async function WeatherPage() {
  const forecast = await fetchForecast();
  const currentMonthIndex = new Date().getMonth();

  return (
    <main className="mx-auto max-w-6xl px-5 py-10 sm:px-6 sm:py-14">
      <header>
        <p className="text-xs uppercase tracking-[0.25em] text-deep/60 sm:text-sm">
          Platja d&apos;Aro · Costa Brava
        </p>
        <h1 className="mt-2 font-display text-3xl font-semibold sm:text-4xl">
          Weather
        </h1>
        <p className="mt-3 max-w-2xl text-base text-deep/70 sm:text-lg">
          A live 7-day forecast for the villa, plus the typical climate for
          each month so you can plan your stay.
        </p>
      </header>

      <section className="mt-10">
        <h2 className="font-display text-xl font-semibold sm:text-2xl">
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

      <section className="mt-14">
        <h2 className="font-display text-xl font-semibold sm:text-2xl">
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
