/**
 * VisionGuard mark: an angular eye inside the four corner brackets of a
 * detection box - the same brackets the backend draws around what it sees.
 *
 * Drawn on a 32-unit grid with integer coordinates so it stays crisp at 16,
 * 24, 32 px and up. Colours come from CSS: the frame uses `currentColor`, the
 * eye uses `--logo-accent` (teal by default; see .logo-mark in styles.css).
 * public/favicon.svg repeats these paths with fixed colours - keep in sync.
 */

/** Four corner brackets, 4 units thick, 10 long, inset 2. */
const FRAME_PATH =
  'M2 2h10v4H6v6H2z M30 2H20v4h6v6h4z M30 30H20v-4h6v-6h4z M2 30h10v-4H6v-6H2z';

/** Flattened hexagon eye with a square pupil cut out (even-odd fill). */
const EYE_PATH = 'M7 16l5-5h8l5 5-5 5h-8z M13 13h6v6h-6z';

interface LogoMarkProps {
  /** Rendered size in px; CSS may override it. */
  size?: number;
  /** One colour throughout (frame and eye both use currentColor). */
  monochrome?: boolean;
  /** Accessible name. Omit when the mark sits next to the written name. */
  title?: string;
  className?: string;
}

export function LogoMark({ size = 28, monochrome = false, title, className }: LogoMarkProps) {
  const classes = ['logo-mark', monochrome && 'logo-mark-mono', className].filter(Boolean).join(' ');
  const labelled = title !== undefined;

  return (
    <svg
      className={classes}
      width={size}
      height={size}
      viewBox="0 0 32 32"
      role={labelled ? 'img' : undefined}
      aria-label={title}
      aria-hidden={labelled ? undefined : true}
      focusable="false"
    >
      <path className="logo-frame" d={FRAME_PATH} />
      <path className="logo-eye" d={EYE_PATH} fillRule="evenodd" />
    </svg>
  );
}

interface WordmarkProps {
  markSize?: number;
}

/**
 * Mark plus the name set in Space Grotesk. The name is real text, so it stays
 * selectable, uses the app's font, and gives the element its accessible name.
 */
export function Wordmark({ markSize = 28 }: WordmarkProps) {
  return (
    <span className="wordmark">
      <LogoMark size={markSize} />
      <span className="wordmark-text">
        <span className="wordmark-strong">Vision</span>
        <span className="wordmark-light">Guard</span>
      </span>
    </span>
  );
}
