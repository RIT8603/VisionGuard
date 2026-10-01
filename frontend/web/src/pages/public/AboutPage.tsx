import { Link } from 'react-router';
import { PATHS } from '../../app/paths';
import { SITE } from '../../app/site';
import { ExternalLink } from '../../components/ui/ExternalLink';

const WHY = [
  {
    title: 'Existing cameras stay useful',
    body: 'VisionGuard reads the RTSP stream that IP cameras and recorders already provide. The investment in cameras is kept; the software around them gets smarter.',
  },
  {
    title: 'From recording to understanding',
    body: 'Instead of footage waiting to be reviewed, every frame is analysed as it arrives: who and what is in view, where they move, and what happened.',
  },
  {
    title: 'Real-time decision support',
    body: 'Events such as a zone breach or a possible fall appear as they happen, so people can respond while it still matters. Automatic alerts and actions are planned next.',
  },
  {
    title: 'Only what you need',
    body: 'Each capability is a module a business chooses. A warehouse and a campus can run the same platform with different intelligence switched on.',
  },
];

const PRINCIPLES = [
  {
    title: 'Modular by configuration',
    body: 'Each capability is an independent module. You enable what you need in one configuration file; disabled modules are never imported and never load a model.',
  },
  {
    title: 'Source-agnostic',
    body: 'Modules never know where frames come from. Moving from a laptop webcam to an RTSP camera changes configuration, not code.',
  },
  {
    title: 'Observe, don’t control',
    body: 'The dashboard can watch a running pipeline but cannot reconfigure or stop it, so a browser tab cannot disturb a camera.',
  },
  {
    title: 'Privacy by default',
    body: 'Face and identity processing are off unless explicitly enabled. Recognition keeps face embeddings, not images, in a store someone builds on purpose.',
  },
];

const STACK = [
  { name: 'Python', role: 'Pipeline and backend' },
  { name: 'OpenCV', role: 'Capture, preprocessing, annotation' },
  { name: 'Ultralytics YOLO', role: 'Detection and pose models' },
  { name: 'FastAPI', role: 'Read-only dashboard API and video stream' },
  { name: 'React + TypeScript', role: 'This web interface' },
];

const LIMITATIONS = [
  'One video source per running process.',
  'No user accounts or authentication yet. Keep the backend on localhost.',
  'The dashboard is read-only; configuration happens in a file.',
  'Fall detection is a heuristic, not a medical device.',
  'PPE, fire and smoke detection need custom-trained models that the project does not ship.',
];

export function AboutPage() {
  return (
    <>
      <title>About · VisionGuard</title>

      <section className="site-page-head">
        <div className="site-container">
          <p className="site-kicker">About</p>
          <h1>Giving existing CCTV cameras a brain.</h1>
          <p className="site-lead">
            Many organisations already have CCTV cameras that mostly just record. VisionGuard turns
            those passive systems into active ones: it understands what the cameras see, analyses it
            in real time and surfaces what matters, without replacing the cameras themselves.
          </p>
        </div>
      </section>

      <section className="site-section" aria-labelledby="why-heading">
        <div className="site-container">
          <p className="site-kicker">Why</p>
          <h2 id="why-heading">Intelligence on top, not a rip-and-replace</h2>
          <ul className="site-grid">
            {WHY.map((item) => (
              <li key={item.title} className="site-card">
                <h3>{item.title}</h3>
                <p>{item.body}</p>
              </li>
            ))}
          </ul>
        </div>
      </section>

      <section className="site-section" aria-labelledby="arch-heading">
        <div className="site-container site-split">
          <div>
            <p className="site-kicker">Architecture</p>
            <h2 id="arch-heading">How the intelligence is added</h2>
            <p>
              Every frame becomes a shared context. Modules run in dependency order and communicate
              only through that context: a detector writes detections, a tracker reads them and
              writes tracks, an analysis module reads tracks and writes events.
            </p>
            <p>
              Adding a capability is one new module plus one registry entry. The pipeline itself
              does not change.
            </p>
          </div>
          <pre className="site-code site-diagram" aria-label="Pipeline diagram">
            <code>{`Camera source
  ↓
Video source abstraction
  ↓
OpenCV vision core
  ↓
Module registry → enabled modules
  ↓
Shared frame context
  ↓
Annotated video · events · dashboard`}</code>
          </pre>
        </div>
      </section>

      <section className="site-section" aria-labelledby="principles-heading">
        <div className="site-container">
          <p className="site-kicker">Principles</p>
          <h2 id="principles-heading">Design principles</h2>
          <ul className="site-grid">
            {PRINCIPLES.map((item) => (
              <li key={item.title} className="site-card">
                <h3>{item.title}</h3>
                <p>{item.body}</p>
              </li>
            ))}
          </ul>
        </div>
      </section>

      <section className="site-section" aria-labelledby="stack-heading">
        <div className="site-container site-split">
          <div>
            <p className="site-kicker">Technology</p>
            <h2 id="stack-heading">Built with</h2>
            <dl className="site-stack">
              {STACK.map((item) => (
                <div key={item.name}>
                  <dt>{item.name}</dt>
                  <dd>{item.role}</dd>
                </div>
              ))}
            </dl>
          </div>
          <div>
            <p className="site-kicker">Current stage</p>
            <h2>Known limitations</h2>
            <ul className="site-list">
              {LIMITATIONS.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </div>
        </div>
      </section>

      <section className="site-section site-cta" aria-labelledby="more-heading">
        <div className="site-container site-cta-inner">
          <div>
            <h2 id="more-heading">Go deeper</h2>
            <p>
              The repository documents the architecture, module specification, roadmap and
              development workflow.
            </p>
          </div>
          <div className="site-actions">
            <ExternalLink href={SITE.repositoryUrl} className="site-btn">
              Repository
            </ExternalLink>
            <Link to={PATHS.portal.dashboard} className="site-btn site-btn-primary">
              Open dashboard
            </Link>
          </div>
        </div>
      </section>
    </>
  );
}
