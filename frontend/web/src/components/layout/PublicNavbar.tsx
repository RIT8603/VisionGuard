import { Link, NavLink } from 'react-router';
import type { NavItem } from '../../app/navigation';
import { PATHS } from '../../app/paths';
import { Wordmark } from '../brand/Logo';

interface PublicNavbarProps {
  items: readonly NavItem[];
}

export function PublicNavbar({ items }: PublicNavbarProps) {
  return (
    <header className="site-nav">
      <div className="site-container site-nav-inner">
        <Link to={PATHS.home} className="site-nav-brand" aria-label="VisionGuard home">
          <Wordmark markSize={26} />
        </Link>
        <nav className="site-nav-links" aria-label="Main">
          <ul>
            {items.map((item) => (
              <li key={item.to}>
                <NavLink
                  to={item.to}
                  end={item.end}
                  className={({ isActive }) => (isActive ? 'site-nav-link is-active' : 'site-nav-link')}
                >
                  {item.label}
                </NavLink>
              </li>
            ))}
          </ul>
        </nav>
        <Link to={PATHS.portal.dashboard} className="site-btn site-btn-primary site-nav-cta">
          Open dashboard
        </Link>
      </div>
    </header>
  );
}
