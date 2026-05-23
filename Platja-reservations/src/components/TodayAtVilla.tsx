import { describeWeatherCode, fetchToday } from "@/lib/weather";

function dayPart(): string {
  const hour = Number(
    new Date().toLocaleString("en-GB", { hour: "2-digit", hourCycle: "h23", timeZone: "Europe/Madrid" }),
  );
  if (hour < 6) return "tonight";
  if (hour < 12) return "this morning";
  if (hour < 18) return "this afternoon";
  return "this evening";
}

function timeAtVilla(): string {
  return new Date().toLocaleTimeString("en-GB", {
    timeZone: "Europe/Madrid",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function FallbackWeather() {
  return null;
}

export async function TodayAtVilla() {
  const today = await fetchToday();
  if (!today) return <FallbackWeather />;

  const { label, icon } = describeWeatherCode(today.weatherCode);
  const rainNote =
    today.precipitationMm > 0.5 ? "Showers expected" : "Dry day ahead";

  const tiles: { label: string; value: string; sub?: string }[] = [
    {
      label: "High / Low",
      value: `${Math.round(today.highC)}° / ${Math.round(today.lowC)}°`,
      sub: "Today",
    },
    {
      label: "Rain",
      value: `${today.precipitationMm.toFixed(1)} mm`,
      sub: rainNote,
    },
    ...(today.seaTempC != null
      ? [
          {
            label: "Sea",
            value: `${Math.round(today.seaTempC)}°`,
            sub: "Surface",
          },
        ]
      : []),
    { label: "Sunrise", value: today.sunrise, sub: "Costa Brava" },
    { label: "Sunset", value: today.sunset, sub: "Golden hour" },
  ];

  return (
    <section className="relative overflow-hidden bg-sand-wash">
      <div aria-hidden className="pointer-events-none absolute inset-0 bg-sun-fade" />
      <div
        aria-hidden
        className="pointer-events-none absolute -top-32 -right-24 h-72 w-72 rounded-full bg-sunset/35 blur-3xl sm:h-[28rem] sm:w-[28rem]"
      />
      <div
        aria-hidden
        className="pointer-events-none absolute -bottom-32 -left-24 h-72 w-72 rounded-full bg-stone/30 blur-3xl sm:h-[28rem] sm:w-[28rem]"
      />

      <div className="relative mx-auto max-w-6xl px-5 pb-16 pt-16 sm:px-6 sm:pb-20 sm:pt-24 lg:pt-28">
        <div className="flex items-center justify-between gap-3">
          <p className="text-[11px] uppercase tracking-[0.3em] text-ink/55 sm:text-xs">
            Villa Mas Nou · Platja d&apos;Aro · Costa Brava
          </p>
          <p className="rounded-full bg-white/70 px-3 py-1 text-[11px] font-medium text-ink/70 ring-1 ring-ink/10 backdrop-blur">
            {timeAtVilla()} local
          </p>
        </div>

        <div className="mt-14 grid gap-12 lg:grid-cols-[1.1fr_1fr] lg:items-end">
          <div>
            <p className="font-display text-sm uppercase tracking-[0.25em] text-ink/55">
              Right now {dayPart()} at the villa
            </p>
            <div className="mt-6 flex items-baseline gap-5">
              <span className="font-display text-[5rem] font-light leading-none tracking-tightish text-ink sm:text-[7rem]">
                {Math.round(today.temperatureC)}°
              </span>
              <span className="text-4xl sm:text-6xl" aria-hidden>
                {icon}
              </span>
            </div>
            <p className="mt-3 font-display text-xl font-light italic text-ink/70 sm:text-2xl">
              {label}
            </p>
            <p className="mt-2 text-sm text-ink/55">
              The pool is open, the terrace is yours.
            </p>
          </div>

          <ul className="grid grid-cols-2 gap-3 self-end sm:grid-cols-3 lg:grid-cols-2 lg:gap-4">
            {tiles.map((t) => (
              <li
                key={t.label}
                className="rounded-3xl bg-white/70 p-4 shadow-soft ring-1 ring-white/60 backdrop-blur-md sm:p-5"
              >
                <p className="text-[10px] uppercase tracking-wider text-ink/55">
                  {t.label}
                </p>
                <p className="mt-1 font-display text-2xl font-semibold text-ink sm:text-3xl">
                  {t.value}
                </p>
                {t.sub && (
                  <p className="mt-0.5 text-xs text-ink/55">{t.sub}</p>
                )}
              </li>
            ))}
          </ul>
        </div>

        <p className="mt-12 text-center text-sm">
          <a href="/weather" className="font-medium text-ink/55 hover:text-ink">
            See the 7-day forecast →
          </a>
        </p>
      </div>
    </section>
  );
}
