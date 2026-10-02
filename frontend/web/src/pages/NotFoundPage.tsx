import { Link, useLocation } from 'react-router';
import { PATHS } from '../app/paths';
import { LogoMark } from '../components/brand/Logo';

export function NotFoundPage() {
  const { pathname } = useLocation();

  return (
    <>
      <title>Page not found · VisionGuard</title>

      <section className="site-notfound">
        <div className="site-container site-notfound-inner">
          <LogoMark size={96} className="site-notfound-mark" />
          <p className="site-notfound-code mono">404</p>
          <h1>Nothing detected here</h1>
          <p className="site-lead">
            There is no page at <code>{pathname}</code>. It may have moved, or the link may be wrong.
          </p>
          <div className="site-actions">
            <Link to={PATHS.home} className="site-btn">
              Back to overview
            </Link>
            <Link to={PATHS.portal.dashboard} className="site-btn site-btn-primary">
              Open dashboard
            </Link>
          </div>
        </div>
      </section>
    </>
  );
}
