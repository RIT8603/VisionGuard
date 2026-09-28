import type { ReactNode } from 'react';

interface PanelProps {
  title: string;
  /** Right-aligned content in the header, e.g. a count. */
  aside?: ReactNode;
  className?: string;
  children: ReactNode;
}

export function Panel({ title, aside, className, children }: PanelProps) {
  return (
    <section className={className ? `panel ${className}` : 'panel'}>
      <header className="panel-head">
        <h2>{title}</h2>
        {aside !== undefined && <span className="panel-aside">{aside}</span>}
      </header>
      <div className="panel-body">{children}</div>
    </section>
  );
}
