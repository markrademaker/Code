import type { Metadata } from "next";
import { format, parseISO } from "date-fns";
import { fetchForecast, describeWeatherCode } from "@/lib/weather";
import { PLATJA_CLIMATE } from "@/lib/climate";
import { SectionMark } from "@/components/Marks";
import { SiteShell } from "@/components/SiteShell";
import { Frost } from "@/components/Frost";

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
  const current = forecast
    ? describeWeatherCode(forecast.current.weatherCode)
    : null;

  return (
    <SiteShell slideshowCount={2}>
      <div className="pt-12 pb-24 sm:pt-16 sm:pb-32">
        <section className="relative mx-auto max-w-7xl px-5 sm:px-8">
          <Frost strength={80} className="p-8 sm:p-12 lg:p-16">
            <SectionMark number="V" label="Costa Brava forecast" />

            <div className="mt-10 grid gap-10 lg:grid-cols-[1.45fr_1fr] lg:items-end lg:gap-20">
              <h1 className="font-display text-4xl font-light leading-[1.05] tracking-tightish text-ink sm:text-5xl lg:text-7xl">
                Sun, <span className="text-terracotta">sea</span>,
                <br />
                and the in-between.
              </h1>
              <p className="text-base leading-relaxed text-ink/75 sm:text-lg lg:pb-3">
                A live seven-day forecast for the villa, plus the typical
                climate for each month so you can plan your stay.
              </p>
            </div>

            {forecast && current ? (
              <>
                {/* Hero "right now" card with the condition photo behind */}
                <div className="relative mt-14 overflow-hidden rounded-3xl shadow-soft ring-1 ring-ink/5">
                  <div
                    aria-hidden
                    className="absolute inset-0 bg-cover bg-center"
                    style={{ backgroundImage: `url(${current.photo})` }}
                  />
                  <div
                    aria-hidden
                    className="absolute inset-0"
                    style={{
                      background:
                        "linear-gradient(110deg, rgba(20,16,12,0.55) 0%, rgba(20,16,12,0.35) 55%, rgba(20,16,12,0.2) 100%)",
                    }}
                  />
                  <div className="relative grid gap-6 p-8 text-whitewash sm:grid-cols-[1.4fr_1fr] sm:p-12 lg:p-16">
                    <div>
                      <p className="text-[11px] uppercase tracking-[0.32em] text-whitewash/80">
                        Right now · Platja d&apos;Aro
                      </p>
                      <p
                        className="mt-6 font-display text-7xl font-light leading-none tracking-tightish sm:text-8xl lg:text-[9rem]"
                        style={{ textShadow: "0 2px 24px rgba(0,0,0,0.35)" }}
                      >
                        {Math.round(forecast.current.temperatureC)}°
                      </p>
                      <p className="mt-3 font-display text-2xl font-light italic text-whitewash/90 sm:text-3xl">
                        {current.label}
                      </p>
                    </div>
                    <dl className="grid grid-cols-2 content-end gap-y-4 self-end text-sm">
                      <Stat
                        label="Today's high"
                        value={`${Math.round(forecast.daily[0]?.highC ?? forecast.current.temperatureC)}°`}
                      />
                      <Stat
                        label="Today's low"
                        value={`${Math.round(forecast.daily[0]?.lowC ?? forecast.current.temperatureC)}°`}
                      />
                      <Stat
                        label="Rain"
                        value={`${(forecast.daily[0]?.precipitationMm ?? 0).toFixed(1)} mm`}
                      />
                      <Stat label="Updated" value="every 30 min" />
                    </dl>
                  </div>
                </div>

                {/* 7-day strip, each day shows the condition photo behind text */}
                <h2 className="mt-14 font-display text-2xl font-light tracking-tightish sm:text-3xl">
                  Next seven days
                </h2>
                <ul className="mt-5 grid gap-3 sm:grid-cols-4 lg:grid-cols-7">
                  {forecast.daily.map((d, i) => {
                    const { label, photo } = describeWeatherCode(d.weatherCode);
                    return (
                      <li
                        key={d.date}
                        className="relative aspect-[3/4] overflow-hidden rounded-2xl shadow-soft ring-1 ring-ink/5"
                      >
                        <div
                          aria-hidden
                          className="absolute inset-0 bg-cover bg-center"
                          style={{ backgroundImage: `url(${photo})` }}
                        />
                        <div
                          aria-hidden
                          className="absolute inset-0"
                          style={{
                            background:
                              "linear-gradient(180deg, rgba(20,16,12,0.15) 0%, rgba(20,16,12,0.65) 100%)",
                          }}
                        />
                        <div className="relative flex h-full flex-col p-4 text-whitewash">
                          <p className="text-[10px] uppercase tracking-[0.25em] text-whitewash/80">
                            {i === 0 ? "Today" : format(parseISO(d.date), "EEE")}
                          </p>
                          <p className="mt-1 text-xs text-whitewash/70">
                            {format(parseISO(d.date), "d MMM")}
                          </p>
                          <div className="mt-auto">
                            <p className="font-display text-3xl font-light leading-none">
                              {Math.round(d.highC)}°
                            </p>
                            <p className="text-xs text-whitewash/75">
                              / {Math.round(d.lowC)}°
                            </p>
                            <p className="mt-2 font-display text-sm italic text-whitewash/90">
                              {label}
                            </p>
                            {d.precipitationMm > 0.2 && (
                              <p className="text-[11px] text-whitewash/75">
                                {d.precipitationMm.toFixed(1)} mm rain
                              </p>
                            )}
                          </div>
                        </div>
                      </li>
                    );
                  })}
                </ul>
                <p className="mt-3 text-xs text-ink/55">
                  Forecast from Open-Meteo, updated every 30 minutes.
                </p>
              </>
            ) : (
              <p className="mt-14 rounded-3xl bg-white p-8 text-ink/70 shadow-soft ring-1 ring-ink/5">
                Forecast unavailable right now. Try again in a bit.
              </p>
            )}

            <section className="mt-20">
              <h2 className="font-display text-2xl font-light tracking-tightish sm:text-3xl">
                Typical weather by month
              </h2>
              <p className="mt-2 max-w-2xl text-sm text-ink/70 sm:text-base">
                Long-term averages for the Costa Brava — useful for picking
                when to visit.
              </p>
              <ul className="mt-6 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                {PLATJA_CLIMATE.map((m, i) => {
                  const isCurrent = i === currentMonthIndex;
                  return (
                    <li
                      key={m.month}
                      className={`rounded-2xl p-5 shadow-soft ring-1 ${
                        isCurrent
                          ? "bg-sea/15 ring-sea/30"
                          : "bg-white ring-ink/5"
                      }`}
                    >
                      <div className="flex items-baseline justify-between">
                        <p className="font-display text-lg font-light">
                          {m.month}
                        </p>
                        {isCurrent && (
                          <span className="rounded-full bg-ocean px-2 py-0.5 text-[10px] font-medium uppercase tracking-wider text-whitewash">
                            Now
                          </span>
                        )}
                      </div>
                      <div className="mt-2 flex gap-4 text-sm text-ink/80">
                        <span>
                          <strong className="font-medium text-ink">
                            {m.highC}°
                          </strong>{" "}
                          / {m.lowC}°
                        </span>
                        <span>Sea {m.seaC}°</span>
                        <span>{m.rainMm} mm</span>
                      </div>
                      <p className="mt-2 text-sm text-ink/70">{m.blurb}</p>
                    </li>
                  );
                })}
              </ul>
            </section>
          </Frost>
        </section>
      </div>
    </SiteShell>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-[10px] uppercase tracking-[0.22em] text-whitewash/65">
        {label}
      </dt>
      <dd className="mt-1 font-display text-2xl font-light text-whitewash">
        {value}
      </dd>
    </div>
  );
}
