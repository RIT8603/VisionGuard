/**
 * Every URL in the app. Link and route with these constants rather than
 * string literals, so a path changes in one place.
 */
export const PATHS = {
  home: '/',
  about: '/about',
  contact: '/contact',
  portal: {
    root: '/portal',
    dashboard: '/portal/dashboard',
    live: '/portal/live',
    events: '/portal/events',
    cameras: '/portal/cameras',
    modules: '/portal/modules',
    alerts: '/portal/alerts',
    faces: '/portal/faces',
    reports: '/portal/reports',
    settings: '/portal/settings',
  },
} as const;
