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

export const PORTAL_NAV: readonly NavItem[] = [
  { label: 'Dashboard', to: PATHS.portal.dashboard },
  { label: 'Modules', to: PATHS.portal.modules },
];
