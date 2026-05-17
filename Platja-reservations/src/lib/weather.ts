export type DailyForecast = {
  date: string;
  highC: number;
  lowC: number;
  precipitationMm: number;
  weatherCode: number;
};

export type CurrentWeather = {
  temperatureC: number;
  weatherCode: number;
  precipitationMm: number;
};

export type Forecast = {
  current: CurrentWeather;
  daily: DailyForecast[];
};

export type Today = {
  temperatureC: number;
  weatherCode: number;
  highC: number;
  lowC: number;
  precipitationMm: number;
  sunrise: string; // HH:mm local
  sunset: string;  // HH:mm local
  seaTempC: number | null;
};

const LAT = 41.817;
const LON = 3.067;

export async function fetchForecast(): Promise<Forecast | null> {
  const url = new URL("https://api.open-meteo.com/v1/forecast");
  url.searchParams.set("latitude", String(LAT));
  url.searchParams.set("longitude", String(LON));
  url.searchParams.set("current", "temperature_2m,weather_code,precipitation");
  url.searchParams.set(
    "daily",
    "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code",
  );
  url.searchParams.set("timezone", "Europe/Madrid");
  url.searchParams.set("forecast_days", "7");

  try {
    const res = await fetch(url, { next: { revalidate: 1800 } });
    if (!res.ok) return null;
    const data = (await res.json()) as {
      current: { temperature_2m: number; weather_code: number; precipitation: number };
      daily: {
        time: string[];
        temperature_2m_max: number[];
        temperature_2m_min: number[];
        precipitation_sum: number[];
        weather_code: number[];
      };
    };
    return {
      current: {
        temperatureC: data.current.temperature_2m,
        weatherCode: data.current.weather_code,
        precipitationMm: data.current.precipitation,
      },
      daily: data.daily.time.map((date, i) => ({
        date,
        highC: data.daily.temperature_2m_max[i],
        lowC: data.daily.temperature_2m_min[i],
        precipitationMm: data.daily.precipitation_sum[i],
        weatherCode: data.daily.weather_code[i],
      })),
    };
  } catch {
    return null;
  }
}

export async function fetchToday(): Promise<Today | null> {
  if (process.env.MOCK_WEATHER === "1") {
    return {
      temperatureC: 27,
      weatherCode: 1,
      highC: 29,
      lowC: 19,
      precipitationMm: 0,
      sunrise: "07:14",
      sunset: "20:38",
      seaTempC: 24,
    };
  }
  const url = new URL("https://api.open-meteo.com/v1/forecast");
  url.searchParams.set("latitude", String(LAT));
  url.searchParams.set("longitude", String(LON));
  url.searchParams.set("current", "temperature_2m,weather_code,precipitation");
  url.searchParams.set(
    "daily",
    "temperature_2m_max,temperature_2m_min,precipitation_sum,sunrise,sunset",
  );
  url.searchParams.set("timezone", "Europe/Madrid");
  url.searchParams.set("forecast_days", "1");

  const marine = new URL("https://marine-api.open-meteo.com/v1/marine");
  marine.searchParams.set("latitude", String(LAT));
  marine.searchParams.set("longitude", String(LON));
  marine.searchParams.set("current", "sea_surface_temperature");
  marine.searchParams.set("timezone", "Europe/Madrid");

  try {
    const [resA, resB] = await Promise.all([
      fetch(url, { next: { revalidate: 1800 } }),
      fetch(marine, { next: { revalidate: 3600 } }).catch(() => null),
    ]);
    if (!resA.ok) return null;
    const data = (await resA.json()) as {
      current: { temperature_2m: number; weather_code: number; precipitation: number };
      daily: {
        temperature_2m_max: number[];
        temperature_2m_min: number[];
        precipitation_sum: number[];
        sunrise: string[];
        sunset: string[];
      };
    };

    let seaTempC: number | null = null;
    if (resB && resB.ok) {
      const marineData = (await resB.json()) as {
        current?: { sea_surface_temperature?: number };
      };
      seaTempC = marineData.current?.sea_surface_temperature ?? null;
    }

    return {
      temperatureC: data.current.temperature_2m,
      weatherCode: data.current.weather_code,
      precipitationMm: data.current.precipitation,
      highC: data.daily.temperature_2m_max[0],
      lowC: data.daily.temperature_2m_min[0],
      sunrise: data.daily.sunrise[0].slice(-5),
      sunset: data.daily.sunset[0].slice(-5),
      seaTempC,
    };
  } catch {
    return null;
  }
}

export function describeWeatherCode(code: number): { label: string; icon: string } {
  if (code === 0) return { label: "Clear", icon: "☀️" };
  if (code <= 2) return { label: "Mostly sunny", icon: "🌤️" };
  if (code === 3) return { label: "Overcast", icon: "☁️" };
  if (code === 45 || code === 48) return { label: "Fog", icon: "🌫️" };
  if (code >= 51 && code <= 57) return { label: "Drizzle", icon: "🌦️" };
  if (code >= 61 && code <= 67) return { label: "Rain", icon: "🌧️" };
  if (code >= 71 && code <= 77) return { label: "Snow", icon: "🌨️" };
  if (code >= 80 && code <= 82) return { label: "Showers", icon: "🌦️" };
  if (code >= 85 && code <= 86) return { label: "Snow showers", icon: "🌨️" };
  if (code >= 95) return { label: "Thunderstorm", icon: "⛈️" };
  return { label: "—", icon: "🌡️" };
}
