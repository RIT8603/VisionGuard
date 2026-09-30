import { useEffect } from 'react';
import { Outlet, useLocation } from 'react-router';
import { PUBLIC_NAV } from '../app/navigation';
import { Footer } from '../components/layout/Footer';
import { PublicNavbar } from '../components/layout/PublicNavbar';

/** Frame for the public site: navbar, page, footer. No polling happens here. */
export function PublicLayout() {
  const { pathname } = useLocation();

  // A new page starts at the top, as it would with full page loads.
  useEffect(() => {
    window.scrollTo(0, 0);
  }, [pathname]);

  return (
    <div className="site">
      <a className="skip-link" href="#site-main">
        Skip to content
      </a>
      <PublicNavbar items={PUBLIC_NAV} />
      <main id="site-main" className="site-main">
        <Outlet />
      </main>
      <Footer items={PUBLIC_NAV} />
    </div>
  );
}
