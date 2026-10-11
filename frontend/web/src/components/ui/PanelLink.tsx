import { Link } from 'react-router';

interface PanelLinkProps {
  to: string;
  children: string;
}

/** Compact "go to detail page" link for a panel header's aside slot. */
export function PanelLink({ to, children }: PanelLinkProps) {
  return (
    <Link to={to} className="panel-link">
      {children}
      <span aria-hidden="true"> →</span>
    </Link>
  );
}
