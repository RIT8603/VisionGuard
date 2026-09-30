import { Link, NavLink } from 'react-router';
import type { NavItem } from '../../app/navigation';
import { PATHS } from '../../app/paths';

interface PortalNavProps {
  items: readonly NavItem[];
}

/**
 * Tab strip under the portal header. Renders whatever PORTAL_NAV lists, so
 * it can grow (or be swapped for a sidebar) without touching the pages.
 * NavLink sets aria-current="page" on the active tab.
 */
export function PortalNav({ items }: PortalNavProps) {
  return (
    <nav className="portal-nav" aria-label="Portal">
      <ul className="portal-tabs">
        {items.map((item) => (
          <li key={item.to}>
            <NavLink
              to={item.to}
              end={item.end}
              className={({ isActive }) => (isActive ? 'portal-tab is-active' : 'portal-tab')}
            >
              {item.label}
            </NavLink>
          </li>
        ))}
      </ul>
      <Link to={PATHS.home} className="portal-nav-aside">
        ← Public site
      </Link>
    </nav>
  );
}
