import { Link } from 'react-router';

interface PanelLinkProps {
  to: string;
  children: string;
}

export function PanelLink({ to, children }: PanelLinkProps) {
  return (
    <Link className="panel-link" to={to}>
      {children}
    </Link>
  );
}
