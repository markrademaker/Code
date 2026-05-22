type IconProps = { className?: string };

export function SunMark({ className = "h-4 w-4" }: IconProps) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      className={className}
      aria-hidden
    >
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v3" />
      <path d="M12 19v3" />
      <path d="M2 12h3" />
      <path d="M19 12h3" />
      <path d="M4.93 4.93l2.12 2.12" />
      <path d="M16.95 16.95l2.12 2.12" />
      <path d="M4.93 19.07l2.12-2.12" />
      <path d="M16.95 7.05l2.12-2.12" />
    </svg>
  );
}

export function WaveMark({ className = "h-3 w-12" }: IconProps) {
  return (
    <svg
      viewBox="0 0 48 12"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.4"
      strokeLinecap="round"
      className={className}
      aria-hidden
    >
      <path d="M2 6 C 6 2, 10 10, 14 6 S 22 2, 26 6 S 34 10, 38 6 S 44 2, 46 6" />
    </svg>
  );
}

export function SectionMark({
  number,
  label,
  className = "",
}: {
  number: string;
  label: string;
  className?: string;
}) {
  return (
    <div
      className={`flex items-center gap-3 text-[11px] uppercase tracking-[0.3em] text-ink/55 ${className}`}
    >
      <span className="font-display text-base italic text-terracotta">
        {number}
      </span>
      <span className="h-px w-10 bg-ink/25" aria-hidden />
      <span>{label}</span>
    </div>
  );
}

export function Divider() {
  return (
    <div className="mx-auto my-2 flex max-w-6xl items-center justify-center gap-3 px-5 py-6 text-terracotta/70 sm:px-6">
      <span className="h-px flex-1 bg-ink/15" aria-hidden />
      <SunMark className="h-3.5 w-3.5" />
      <span className="h-px flex-1 bg-ink/15" aria-hidden />
    </div>
  );
}
