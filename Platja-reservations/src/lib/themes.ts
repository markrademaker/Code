/**
 * Available themes. The actual colour values live in
 * src/app/globals.css under `:root` (default) and
 * `[data-theme="<name>"]` selectors. To switch theme at runtime:
 *
 *   document.documentElement.dataset.theme = "seaside";
 *
 * Or render once on the server by setting it on the <html> element
 * in src/app/layout.tsx.
 */
export const THEMES = [
  {
    name: "default",
    label: "Sand & terracotta",
    description: "The Costa Brava palette — sun-bleached sand with warm ink.",
  },
  {
    name: "seaside",
    label: "Seaside",
    description: "Cooler Mediterranean blues with the same terracotta accent.",
  },
  {
    name: "midnight",
    label: "Midnight",
    description: "Dark mode with warm ink-on-cream contrast inverted.",
  },
] as const;

export type ThemeName = (typeof THEMES)[number]["name"];
