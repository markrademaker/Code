export type SlideshowImage = {
  src: string;
  caption: string;
};

// Costa Brava / Mediterranean stock from Unsplash — swap for real villa
// photos by replacing each entry's `src`.
export const DEFAULT_VILLA_IMAGES: SlideshowImage[] = [
  {
    src: "https://images.unsplash.com/photo-1582719508461-905c673771fd?w=2400&q=80&auto=format&fit=crop",
    caption: "The pool, at dusk",
  },
  {
    src: "https://images.unsplash.com/photo-1499793983690-e29da59ef1c2?w=2400&q=80&auto=format&fit=crop",
    caption: "White walls, blue shutters",
  },
  {
    src: "https://images.unsplash.com/photo-1540541338287-41700207dee6?w=2400&q=80&auto=format&fit=crop",
    caption: "Sun terrace",
  },
  {
    src: "https://images.unsplash.com/photo-1505228395891-9a51e7e86bf6?w=2400&q=80&auto=format&fit=crop",
    caption: "The bay, below",
  },
  {
    src: "https://images.unsplash.com/photo-1564501049412-61c2a3083791?w=2400&q=80&auto=format&fit=crop",
    caption: "Bougainvillea path",
  },
  {
    src: "https://images.unsplash.com/photo-1568605114967-8130f3a36994?w=2400&q=80&auto=format&fit=crop",
    caption: "Evening light",
  },
];
