import { Link } from 'react-router';
import type { NavItem } from '../../app/navigation';
import { PATHS } from '../../app/paths';
import { SITE } from '../../app/site';
import { Wordmark } from '../brand/Logo';
import { ExternalLink } from '../ui/ExternalLink';

interface FooterProps {
  items: readonly NavItem[];
}

export function Footer({ items }: FooterProps) {
  return (
    <footer className="site-footer">
      <div className="site-container site-footer-inner">
        <div className="site-footer-brand">
          <Wordmark markSize={22} />
          <p>Modular real-time computer vision. Development prototype.</p>
        </div>
        <nav aria-label="Footer">
          <ul className="site-footer-links">
            {items.map((item) => (
              <li key={item.to}>
                <Link to={item.to}>{item.label}</Link>
              </li>
            ))}
            <li>
              <Link to={PATHS.portal.dashboard}>Dashboard</Link>
            </li>
            <li>
              <ExternalLink href={SITE.repositoryUrl}>GitHub</ExternalLink>
            </li>
          </ul>
        </nav>
      </div>
    </footer>
  );
}
