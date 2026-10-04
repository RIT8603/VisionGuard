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
    modules: '/portal/modules',
    cameras: '/portal/cameras',
    /** Route pattern; build links with cameraPath(). */
    cameraDetail: '/portal/cameras/:cameraId',
  },
} as const;

/** URL of one camera's detail page. */
export function cameraPath(cameraId: string): string {
  return `${PATHS.portal.cameras}/${encodeURIComponent(cameraId)}`;
}
