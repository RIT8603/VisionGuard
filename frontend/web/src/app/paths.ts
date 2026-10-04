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
    cameras: '/portal/cameras',
    modules: '/portal/modules',
  },
} as const;
