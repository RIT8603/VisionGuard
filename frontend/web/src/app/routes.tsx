import { Navigate, Route, Routes } from 'react-router';
import { PortalLayout } from '../layouts/PortalLayout';
import { PublicLayout } from '../layouts/PublicLayout';
import { NotFoundPage } from '../pages/NotFoundPage';
import { AlertsPage } from '../pages/portal/AlertsPage';
import { CamerasPage } from '../pages/portal/CamerasPage';
import { DashboardPage } from '../pages/portal/DashboardPage';
import { ModulesPage } from '../pages/portal/ModulesPage';
import { SettingsPage } from '../pages/portal/SettingsPage';
import { AboutPage } from '../pages/public/AboutPage';
import { ContactPage } from '../pages/public/ContactPage';
import { LandingPage } from '../pages/public/LandingPage';
import { PATHS } from './paths';

/**
 * The complete route table.
 *
 * Adding a portal page: create it under pages/portal/, add its path to
 * paths.ts, a <Route> below, and an entry in PORTAL_NAV (navigation.ts).
 */
export function AppRoutes() {
  return (
    <Routes>
      <Route element={<PublicLayout />}>
        <Route index element={<LandingPage />} />
        <Route path={PATHS.about} element={<AboutPage />} />
        <Route path={PATHS.contact} element={<ContactPage />} />
        <Route path="*" element={<NotFoundPage />} />
      </Route>

      <Route path={PATHS.portal.root} element={<PortalLayout />}>
        <Route index element={<Navigate to={PATHS.portal.dashboard} replace />} />
        <Route path={PATHS.portal.dashboard} element={<DashboardPage />} />
        <Route path={PATHS.portal.cameras} element={<CamerasPage />} />
        <Route path={PATHS.portal.modules} element={<ModulesPage />} />
        <Route path={PATHS.portal.alerts} element={<AlertsPage />} />
        <Route path={PATHS.portal.settings} element={<SettingsPage />} />
      </Route>
      {/* No portal catch-all: unknown /portal/* URLs fall through to the public
          404 above, which keeps the URL the visitor actually typed. */}
    </Routes>
  );
}
