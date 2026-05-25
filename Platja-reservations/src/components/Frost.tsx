import type { CSSProperties, ElementType, ReactNode } from "react";

export function Frost({
  children,
  className = "",
  strength = 80,
  as: As = "section" as ElementType,
  id,
  style,
}: {
  children: ReactNode;
  className?: string;
  strength?: number;
  as?: ElementType;
  id?: string;
  style?: CSSProperties;
}) {
  const a = Math.max(0, Math.min(100, strength)) / 100;
  return (
    <As
      id={id}
      className={`relative overflow-hidden rounded-[2rem] ring-1 ring-white/40 ${className}`}
      style={{
        background: `rgba(250,246,236,${a.toFixed(3)})`,
        backdropFilter: "blur(22px) saturate(150%)",
        WebkitBackdropFilter: "blur(22px) saturate(150%)",
        boxShadow:
          "0 1px 0 rgba(255,255,255,0.5) inset, 0 20px 60px -20px rgba(28,22,16,0.35)",
        ...style,
      }}
    >
      {children}
    </As>
  );
}
