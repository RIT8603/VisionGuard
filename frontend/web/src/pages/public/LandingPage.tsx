import { Link } from 'react-router';
import { PATHS } from '../../app/paths';

/*
 * Content rule: "What works today" lists only modules the repository README
 * marks as working. Anything that also needs model files the project does not
 * ship is labelled as such. The roadmap lists undelivered items from
 * docs/ROADMAP.md and is styled so it cannot be mistaken for current features.
 */

type Availability = 'available' | 'setup';

interface Capability {
  title: string;
  body: string;
  detail: string;
  availability: Availability;
}

const AVAILABILITY_LABEL: Record<Availability, string> = {
  available: 'Available',
  setup: 'Needs model files',
};

const CAPABILITIES: Capability[] = [
  {
    title: 'Any video source',
    body: 'Webcam for development, video files for evaluation, RTSP IP cameras for deployment. Switching is a configuration change.',
    detail: 'webcam · file · rtsp',
    availability: 'available',
  },
  {
    title: 'Detection',
    body: 'People, vehicles and the 80 everyday object classes of the COCO dataset, using a pretrained YOLO model.',
    detail: 'person · vehicle · object',
    availability: 'available',
  },
  {
    title: 'Tracking',
    body: 'Persistent IDs follow each person, vehicle or object from frame to frame.',
    detail: 'person · vehicle · object tracks',
    availability: 'available',
  },
  {
    title: 'Events',
    body: 'Restricted-zone entry, exit and dwell; fall detection from body pose; live crowd counts.',
    detail: 'zones · falls · crowd count',
    availability: 'available',
  },
  {
    title: 'Live monitoring',
    body: 'A browser dashboard with the annotated video, per-module performance and a running event log.',
    detail: 'read-only · local network',
    availability: 'available',
  },
  {
    title: 'Identity (opt-in)',
    body: 'Face recognition, attendance and unknown-person alerts. Off by default, and they need face models plus an identity store built deliberately.',
    detail: 'face · attendance · unknown person',
    availability: 'setup',
  },
];

/** Before/after pairs. "After" lines describe only what works today. */
const SHIFT = [
  {
    before: 'Records footage for someone to review later',
    after: 'Analyses every frame as it arrives',
  },
  {
    before: 'Sees pixels',
    after: 'Recognises people, vehicles and objects, and follows them across frames',
  },
  {
    before: 'Needs a person watching to notice anything',
    after: 'Raises events: zone entries, dwell time, possible falls',
  },
  {
    before: 'Smarter coverage usually means new hardware',
    after: 'Works with the RTSP cameras already installed',
  },
];

const STEPS = [
  { title: 'Connect a source', body: 'Point VisionGuard at a webcam, a recording or an RTSP camera.' },
  { title: 'Choose modules', body: 'Enable only the analysis you need. Disabled modules never load a model.' },
  { title: 'Analyse in real time', body: 'Detection feeds tracking; tracking feeds events, frame by frame.' },
  { title: 'Watch the results', body: 'Annotated video, telemetry and events appear on the dashboard.' },
];

const APPLICATIONS = [
  { title: 'Restricted areas', body: 'Flag people entering or lingering in a zone you define.', module: 'restricted_zone' },
  { title: 'Falls', body: 'Surface possible falls for a person to check. A heuristic, not a medical device.', module: 'fall_detection' },
  { title: 'Occupancy', body: 'Keep a live count of people in view.', module: 'crowd_detection' },
  { title: 'Vehicle movement', body: 'Detect and track cars, trucks, buses and bikes.', module: 'vehicle_tracking' },
];

/** Undelivered items from docs/ROADMAP.md. */
const ROADMAP = [
  { phase: 'Phase 6', title: 'PPE, fire and smoke detection', body: 'Detector code exists, but no suitable model ships with the project yet.' },
  { phase: 'Phase 9', title: 'PPE-violation and crowd events', body: 'Higher-level events built on the detectors above.' },
  { phase: 'Phase 10', title: 'Full dashboard', body: 'Camera management, module selection, alerts and configuration from the browser.' },
  { phase: 'Phase 11', title: 'Cloud deployment', body: 'Optional cloud services for storage, notifications and monitoring.' },
  { phase: 'Phase 12', title: 'Agentic vision', body: 'Only once the deterministic pipeline is stable.' },
];

export function LandingPage() {
  return (
    <>
      <title>VisionGuard — Give your existing CCTV cameras a brain</title>

      <section className="site-hero">
        <div className="site-container site-hero-inner">
          <div className="site-hero-copy">
            <p className="site-kicker">AI for existing CCTV</p>
            <h1>VisionGuard gives existing CCTV cameras a brain.</h1>
            <p className="site-lead">
              Most cameras only record, and someone reviews the footage after the fact. VisionGuard adds
              intelligence on top of the cameras you already have, so they understand the scene,
              analyse what is happening in real time and raise events the moment something needs
              attention.
            </p>
            <div className="site-actions">
              <Link to={PATHS.portal.dashboard} className="site-btn site-btn-primary">
                Open live dashboard
              </Link>
              <Link to={PATHS.portal.modules} className="site-btn">
                Explore modules
              </Link>
            </div>
          </div>

          <ol className="site-flow" aria-label="Processing pipeline">
            <li>
              <span className="site-flow-step mono">01</span>
              <span className="site-flow-title">Video source</span>
              <span className="site-flow-detail mono">webcam · file · rtsp</span>
            </li>
            <li>
              <span className="site-flow-step mono">02</span>
              <span className="site-flow-title">OpenCV vision core</span>
              <span className="site-flow-detail mono">capture · resize · timing</span>
            </li>
            <li>
              <span className="site-flow-step mono">03</span>
              <span className="site-flow-title">Enabled AI modules</span>
              <span className="site-flow-detail mono">detect → track → analyse</span>
            </li>
            <li>
              <span className="site-flow-step mono">04</span>
              <span className="site-flow-title">Results</span>
              <span className="site-flow-detail mono">detections · tracks · events</span>
            </li>
            <li>
              <span className="site-flow-step mono">05</span>
              <span className="site-flow-title">Live dashboard</span>
              <span className="site-flow-detail mono">video · telemetry · event log</span>
            </li>
          </ol>
        </div>
      </section>

      <div className="site-container">
        <div className="notice notice-warn site-stage" role="note">
          <span className="notice-word">Prototype</span>
          <div className="notice-body">
            VisionGuard is in active development and runs locally. Today that means one camera per
            process, a read-only dashboard and no user accounts. It is not yet a commercial product.
          </div>
        </div>
      </div>

      <section className="site-section" aria-labelledby="shift-heading">
        <div className="site-container">
          <p className="site-kicker">From recording to understanding</p>
          <h2 id="shift-heading">Keep your cameras. Add the intelligence.</h2>
          <div className="site-compare">
            <div className="site-compare-col">
              <h3>Passive CCTV</h3>
              <ul className="site-list">
                {SHIFT.map((row) => (
                  <li key={row.before}>{row.before}</li>
                ))}
              </ul>
            </div>
            <div className="site-compare-col site-compare-after">
              <h3>With VisionGuard</h3>
              <ul className="site-list">
                {SHIFT.map((row) => (
                  <li key={row.after}>{row.after}</li>
                ))}
              </ul>
            </div>
          </div>
          <p className="site-section-note">
            Connects to IP cameras and recorders that provide an RTSP stream. Automatic alerts and
            actions beyond the dashboard are on the roadmap.
          </p>
        </div>
      </section>

      <section className="site-section" aria-labelledby="works-heading">
        <div className="site-container">
          <p className="site-kicker">Current capabilities</p>
          <h2 id="works-heading">What works today</h2>
          <ul className="site-grid">
            {CAPABILITIES.map((item) => (
              <li key={item.title} className="site-card">
                <span className={`site-tag site-tag-${item.availability}`}>
                  {AVAILABILITY_LABEL[item.availability]}
                </span>
                <h3>{item.title}</h3>
                <p>{item.body}</p>
                <p className="site-card-detail mono">{item.detail}</p>
              </li>
            ))}
          </ul>
        </div>
      </section>

      <section className="site-section" aria-labelledby="how-heading">
        <div className="site-container">
          <p className="site-kicker">Workflow</p>
          <h2 id="how-heading">How it works</h2>
          <ol className="site-steps">
            {STEPS.map((step, index) => (
              <li key={step.title}>
                <span className="site-step-number mono">{String(index + 1).padStart(2, '0')}</span>
                <h3>{step.title}</h3>
                <p>{step.body}</p>
              </li>
            ))}
          </ol>
        </div>
      </section>

      <section className="site-section" aria-labelledby="apply-heading">
        <div className="site-container">
          <p className="site-kicker">Potential applications</p>
          <h2 id="apply-heading">Where these capabilities could help</h2>
          <p className="site-section-lead">
            Examples built only from the modules above. Any real deployment needs testing on its own
            cameras and conditions.
          </p>
          <ul className="site-grid site-grid-compact">
            {APPLICATIONS.map((item) => (
              <li key={item.title} className="site-card">
                <h3>{item.title}</h3>
                <p>{item.body}</p>
                <p className="site-card-detail mono">{item.module}</p>
              </li>
            ))}
          </ul>
        </div>
      </section>

      <section className="site-section" aria-labelledby="roadmap-heading">
        <div className="site-container">
          <div className="site-roadmap">
            <p className="site-kicker site-kicker-muted">Roadmap</p>
            <h2 id="roadmap-heading">Planned — not available yet</h2>
            <p className="site-section-lead">
              From the project roadmap. None of these can be used today, and plans may change.
            </p>
            <ul className="site-roadmap-list">
              {ROADMAP.map((item) => (
                <li key={item.title}>
                  <span className="site-tag site-tag-planned">Planned</span>
                  <span className="site-roadmap-phase mono">{item.phase}</span>
                  <h3>{item.title}</h3>
                  <p>{item.body}</p>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </section>

      <section className="site-section site-cta" aria-labelledby="run-heading">
        <div className="site-container site-cta-inner">
          <div>
            <h2 id="run-heading">See it running</h2>
            <p>
              Start the backend with its dashboard enabled, then open the portal. It works with a
              webcam or a recorded video file.
            </p>
            <pre className="site-code">
              <code>python -m backend.main --no-window --dashboard</code>
            </pre>
          </div>
          <Link to={PATHS.portal.dashboard} className="site-btn site-btn-primary">
            Open live dashboard
          </Link>
        </div>
      </section>
    </>
  );
}
