import { OFFICIAL_CONTACTS, SITE } from '../../app/site';
import { ExternalLink } from '../../components/ui/ExternalLink';

export function ContactPage() {
  return (
    <>
      <title>Contact · VisionGuard</title>

      <section className="site-page-head">
        <div className="site-container">
          <p className="site-kicker">Contact</p>
          <h1>Get in touch</h1>
          <p className="site-lead">
            VisionGuard is hosted on GitHub. Questions, bug reports and feature ideas can all go
            through the repository.
          </p>
        </div>
      </section>

      <section className="site-section">
        <div className="site-container">
          <ul className="site-grid">
            <li className="site-card">
              <h2 className="site-card-title">Project repository</h2>
              <p>Source code, documentation and setup instructions.</p>
              <ExternalLink href={SITE.repositoryUrl} className="site-btn">
                View on GitHub
              </ExternalLink>
            </li>
            <li className="site-card">
              <h2 className="site-card-title">Issues and requests</h2>
              <p>Report a problem or suggest a feature through the issue tracker.</p>
              <ExternalLink href={SITE.issuesUrl} className="site-btn">
                Open the issue tracker
              </ExternalLink>
            </li>
            <li className="site-card">
              <h2 className="site-card-title">Official contact</h2>
              {OFFICIAL_CONTACTS.length === 0 ? (
                <p>
                  Official contact details have not been published yet. Please use the issue tracker
                  in the meantime.
                </p>
              ) : (
                <dl className="site-stack">
                  {OFFICIAL_CONTACTS.map((channel) => (
                    <div key={channel.label}>
                      <dt>{channel.label}</dt>
                      <dd>
                        <a href={channel.href}>{channel.value}</a>
                      </dd>
                    </div>
                  ))}
                </dl>
              )}
            </li>
          </ul>
        </div>
      </section>
    </>
  );
}
