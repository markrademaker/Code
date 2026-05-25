import type { ReactNode } from "react";
import { VillaSlideshow } from "@/components/VillaSlideshow";
import {
  DEFAULT_VILLA_IMAGES,
  type SlideshowImage,
} from "@/lib/villa-images";

export function SiteShell({
  children,
  slideshow = true,
  slideshowImages,
  slideshowCount,
  showFooter = true,
}: {
  children: ReactNode;
  slideshow?: boolean;
  slideshowImages?: SlideshowImage[];
  slideshowCount?: number;
  showFooter?: boolean;
}) {
  const images =
    slideshowImages ??
    DEFAULT_VILLA_IMAGES.slice(
      0,
      Math.max(2, Math.min(DEFAULT_VILLA_IMAGES.length, slideshowCount ?? DEFAULT_VILLA_IMAGES.length)),
    );
  return (
    <>
      {slideshow && (
        <VillaSlideshow mode="kenburns" darkOverlay={32} images={images} />
      )}
      <div className="relative" style={{ zIndex: 1 }}>
        <main className="relative">
          {children}
          {showFooter && <SiteFooter />}
        </main>
      </div>
    </>
  );
}

export function SiteFooter() {
  return (
    <footer
      className="relative mt-32 border-t border-whitewash/25 py-16 text-center"
      style={{
        background: "rgba(28,22,16,0.55)",
        backdropFilter: "blur(14px) saturate(150%)",
        WebkitBackdropFilter: "blur(14px) saturate(150%)",
      }}
    >
      <p className="font-display text-3xl italic text-whitewash">
        Villa Mas Nou
      </p>
      <p className="mt-2 text-[11px] uppercase tracking-[0.3em] text-whitewash/70">
        Platja d&apos;Aro · Costa Brava
      </p>
      <p className="mt-6 text-xs text-whitewash/55">
        © {new Date().getFullYear()}
      </p>
    </footer>
  );
}
