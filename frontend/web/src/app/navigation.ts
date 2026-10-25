import { PATHS } from './paths';

/**
 * Navigation menus, kept apart from the components that render them.
 * Adding a portal page = add its route in routes.tsx and one entry here.
 */
export interface NavItem {
  label: string;
  to: string;
  /** Match the path exactly (for "/", which every URL starts with). */
  end?: boolean;
}

export const PUBLIC_NAV: readonly NavItem[] = [
  { label: 'Overview', to: PATHS.home, end: true },
  { label: 'About', to: PATHS.about },
  { label: 'Contact', to: PATHS.contact },
];

/**
 * Only pages backed by real data are listed. Zones, Reports and Settings
 * join once the backend can support them: add the path, the route and one
 * entry here.
 */
export const PORTAL_NAV: readonly NavItem[] = [
  { label: 'Dashboard', to: PATHS.portal.dashboard },
  { label: 'Live Monitoring', to: PATHS.portal.live },
  { label: 'Cameras', to: PATHS.portal.cameras },
  { label: 'Events & Alerts', to: PATHS.portal.events },
  { label: 'Modules', to: PATHS.portal.modules },
  { label: 'Face Registry', to: PATHS.portal.faces },
  { label: 'Alerts', to: PATHS.portal.alerts },
  { label: 'Settings', to: PATHS.portal.settings },
];
