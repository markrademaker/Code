"use client";

import { useEffect, useState } from "react";
import {
  DEFAULT_VILLA_IMAGES,
  type SlideshowImage,
} from "@/lib/villa-images";

export { DEFAULT_VILLA_IMAGES };
export type { SlideshowImage };

type Mode = "fade" | "kenburns" | "parallax" | "snap";

function useScrollProgress(): number {
  const [p, setP] = useState(0);
  useEffect(() => {
    let raf = 0;
    const update = () => {
      raf = 0;
      const max = document.documentElement.scrollHeight - window.innerHeight;
      setP(max > 0 ? Math.min(1, Math.max(0, window.scrollY / max)) : 0);
    };
    const onScroll = () => {
      if (!raf) raf = requestAnimationFrame(update);
    };
    update();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", update);
    return () => {
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", update);
      if (raf) cancelAnimationFrame(raf);
    };
  }, []);
  return p;
}

const triangular = (t: number): number => Math.max(0, 1 - Math.abs(t));
const smoothstep = (x: number): number => {
  const v = Math.max(0, Math.min(1, x));
  return v * v * (3 - 2 * v);
};

export function VillaSlideshow({
  images = DEFAULT_VILLA_IMAGES,
  mode = "kenburns",
  darkOverlay = 28,
  speed = 1,
}: {
  images?: SlideshowImage[];
  mode?: Mode;
  darkOverlay?: number;
  speed?: number;
}) {
  const p = useScrollProgress();
  const N = images.length;
  const eased = speed === 1 ? p : Math.pow(p, 1 / speed);
  const position = eased * Math.max(1, N - 1);
  const overlayAlpha = Math.max(0, Math.min(100, darkOverlay)) / 100;

  return (
    <div
      aria-hidden
      className="pointer-events-none fixed inset-0 z-0 overflow-hidden"
      style={{ background: "rgb(28,22,16)" }}
    >
      {images.map((img, i) => {
        const t = position - i;

        let opacity: number;
        if (mode === "snap") {
          const ramp = 0.18;
          opacity = smoothstep((ramp - Math.abs(t)) / ramp);
        } else {
          opacity = smoothstep(triangular(t));
        }

        let transform = "translate3d(0,0,0) scale(1.04)";
        if (mode === "kenburns") {
          const clamped = Math.max(-1.2, Math.min(1.2, t));
          const through = (clamped + 1) / 2;
          const scale = 1.12 - 0.12 * through + 0.08 * through * through;
          const dx = (i % 2 ? 1 : -1) * (through - 0.5) * 30;
          const dy = (through - 0.5) * 22;
          transform = `translate3d(${dx.toFixed(1)}px, ${dy.toFixed(1)}px, 0) scale(${scale.toFixed(4)})`;
        } else if (mode === "parallax") {
          const dy = -t * 90;
          transform = `translate3d(0, ${dy.toFixed(1)}px, 0) scale(1.08)`;
        } else if (mode === "fade") {
          transform = "translate3d(0,0,0) scale(1.05)";
        }

        return (
          <div
            key={img.src}
            className="absolute inset-0"
            style={{ opacity, willChange: "opacity, transform" }}
          >
            <img
              src={img.src}
              alt=""
              loading={i < 2 ? "eager" : "lazy"}
              draggable={false}
              className="absolute inset-0 h-full w-full object-cover object-center"
              style={{ transform, willChange: "transform" }}
            />
          </div>
        );
      })}

      <div
        className="absolute inset-0"
        style={{
          background: `linear-gradient(180deg, rgba(28,22,16,${(overlayAlpha * 0.55).toFixed(3)}) 0%, rgba(28,22,16,${overlayAlpha.toFixed(3)}) 100%)`,
        }}
      />

      <SlideIndicator
        position={position}
        count={N}
        captions={images.map((i) => i.caption)}
      />
    </div>
  );
}

function SlideIndicator({
  position,
  count,
  captions,
}: {
  position: number;
  count: number;
  captions: string[];
}) {
  const idx = Math.max(0, Math.min(count - 1, Math.round(position)));
  return (
    <div
      className="pointer-events-none fixed bottom-6 right-6 z-20 flex items-center gap-3 rounded-full border px-4 py-2 backdrop-blur-md"
      style={{
        background: "rgba(250,246,236,0.78)",
        borderColor: "rgba(250,246,236,0.6)",
        boxShadow: "0 8px 30px -10px rgba(0,0,0,0.35)",
      }}
    >
      <span className="font-display text-base italic text-terracotta">
        {String(idx + 1).padStart(2, "0")} / {String(count).padStart(2, "0")}
      </span>
      <span className="flex gap-1">
        {Array.from({ length: count }).map((_, i) => {
          const active = i === idx;
          return (
            <span
              key={i}
              className="block h-0.5 rounded transition-all"
              style={{
                width: active ? 18 : 6,
                background: active ? "rgb(201 123 91)" : "rgba(61,47,36,0.25)",
              }}
            />
          );
        })}
      </span>
      <span className="hidden max-w-[14rem] truncate font-display text-sm italic text-ink/65 sm:inline">
        {captions[idx]}
      </span>
    </div>
  );
}
