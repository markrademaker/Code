import type { Config } from "tailwindcss";

const tokenColors = [
  "whitewash",
  "sand",
  "stone",
  "sea",
  "ocean",
  "terracotta",
  "sunset",
  "olive",
  "ink",
  "deep",
] as const;

const colors: Record<string, string> = {};
for (const name of tokenColors) {
  colors[name] = `rgb(var(--color-${name}) / <alpha-value>)`;
}

const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors,
      fontFamily: {
        display: ["var(--font-display)", "Georgia", "ui-serif", "serif"],
        sans: ["var(--font-sans)", "ui-sans-serif", "system-ui", "sans-serif"],
      },
      boxShadow: {
        soft: "0 2px 30px -10px rgb(var(--color-ink) / 0.10)",
        glow: "0 8px 40px -8px rgb(var(--color-ink) / 0.30)",
      },
      backgroundImage: {
        "sun-fade":
          "radial-gradient(ellipse at top, rgb(var(--color-sunset) / 0.25), transparent 60%), radial-gradient(ellipse at bottom right, rgb(var(--color-sea) / 0.20), transparent 55%)",
        "sand-wash":
          "linear-gradient(180deg, rgb(var(--color-whitewash)) 0%, rgb(var(--color-sand) / 0.7) 100%)",
      },
      letterSpacing: {
        tightish: "-0.015em",
      },
    },
  },
  plugins: [],
};

export default config;
