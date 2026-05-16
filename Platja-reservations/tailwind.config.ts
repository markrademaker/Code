import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        whitewash: "#faf6ec",
        sand: "#ede0c3",
        stone: "#c9b896",
        sea: "#4fa8a8",
        ocean: "#1d6e6e",
        terracotta: "#c97b5b",
        sunset: "#e8a87c",
        olive: "#8a9a5b",
        ink: "#3d2f24",
        deep: "#3d2f24",
      },
      fontFamily: {
        display: ["Georgia", "ui-serif", "serif"],
      },
      boxShadow: {
        soft: "0 2px 30px -10px rgba(61,47,36,0.10)",
        glow: "0 8px 40px -8px rgba(79,168,168,0.35)",
      },
      backgroundImage: {
        "sun-fade":
          "radial-gradient(ellipse at top, rgba(232,168,124,0.25), transparent 60%), radial-gradient(ellipse at bottom right, rgba(79,168,168,0.18), transparent 55%)",
        "sand-wash":
          "linear-gradient(180deg, #faf6ec 0%, #f3eadb 100%)",
      },
    },
  },
  plugins: [],
};

export default config;
