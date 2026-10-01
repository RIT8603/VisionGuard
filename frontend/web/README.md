# VisionGuard web frontend

React + Vite + TypeScript frontend for VisionGuard, in two parts:

- **Public site**: Landing, About and Contact pages describing the project.
- **Portal**: the live monitoring dashboard for a running VisionGuard pipeline,
  with annotated video, run status, per-frame detections and tracks, the event
  log and performance telemetry.

The portal is a read-only client of the dashboard API in
[`backend/api/server.py`](../../backend/api/server.py). Nothing here changes the
backend, the vision pipeline or the legacy page in
[`frontend/static/`](../static/), which keeps working as before.

## Requirements

- Node.js 20 or newer (npm comes with it)
- For the portal: the VisionGuard backend running with its dashboard enabled
  (see [Connecting to the backend](#connecting-to-the-backend)). The public
  pages work without it.

## Running locally

```bash
cd frontend/web
npm install
npm run dev
```

Open <http://localhost:5173> for the public site, or go straight to the
dashboard at <http://localhost:5173/portal/dashboard>.

| Script | What it does |
|---|---|
| `npm run dev` | Vite dev server with the backend proxy |
| `npm run typecheck` | TypeScript only, no output |
| `npm run build` | Type-check, then production build into `dist/` |
| `npm run preview` | Serve `dist/` locally, with the same proxy |

## Routes

| Path | Layout | Page |
|---|---|---|
| `/` | Public | Landing |
| `/about` | Public | About |
| `/contact` | Public | Contact |
| `/portal` | Portal | Redirects to `/portal/dashboard` |
| `/portal/dashboard` | Portal | Live monitoring dashboard |
| `/portal/modules` | Portal | Module selection (review and select capabilities; not saved) |
| anything else | Public | Branded 404 (keeps the URL that was typed) |

All paths are defined once in [`src/app/paths.ts`](src/app/paths.ts); link with
`PATHS.*` rather than string literals.

## Folder structure

```
src/
  main.tsx                 entry: fonts, global styles, <App />
  config.ts                poll intervals, timeouts, thresholds
  app/
    App.tsx                BrowserRouter + routes
    routes.tsx             the complete route table
    paths.ts               every URL, as constants
    navigation.ts          PUBLIC_NAV and PORTAL_NAV menu entries
    site.ts                repository links and official contact channels
  layouts/
    PublicLayout.tsx       navbar + page + footer
    PortalLayout.tsx       PipelineProvider + portal header + tabs + notice + page
  pages/
    public/                LandingPage, AboutPage, ContactPage
    portal/                DashboardPage, ModulesPage
    NotFoundPage.tsx
  content/
    moduleCatalog.ts       customer-facing module list with status and backend keys
  components/
    ui/                    generic: Panel, ExternalLink
    brand/                 Logo (LogoMark, Wordmark)
    layout/                PublicNavbar, Footer, PortalHeader, PortalNav
    modules/               ModuleCard, SelectionSummary
    pipeline/              dashboard panels: LiveStream, StatsPanel, StatusPanel,
                           ModulesPanel, FrameResults, EventFeed,
                           ConnectionNotice, StateBadge
  api/                     typed models, shape guards, fetch client, endpoints
  hooks/                   usePipelineStatus, useEventFeed, usePageVisible
  state/
    viewState.ts           derives loading/waiting/live/stale/stopped/offline
    PipelineProvider.tsx   shares one set of polling loops across the portal
  lib/format.ts            number, time and event formatting
  styles/
    index.css              design tokens, base styles, portal and dashboard
    public.css             public site only (every class starts with `site-`)
```

Rule of thumb: **pages arrange components and read data; they never call
`fetch`**. Network access lives in `api/`, polling in `hooks/`, shared state in
`state/`.

## Routing and layouts

React Router 7 in its declarative mode (`BrowserRouter`, `Routes`, `Route`,
`Outlet`, `NavLink`). Data loaders and framework mode are deliberately unused;
the polling hooks already handle data.

- **`PublicLayout`**: sticky navbar with an "Open dashboard" button, the page,
  and a footer. It never polls the backend.
- **`PortalLayout`**: wraps everything in `PipelineProvider`, then renders the
  status header, the tab strip, the connection notice (offline, stale,
  stopped) and the page. Leaving the portal unmounts it, which stops polling
  and closes the video stream.

Navigation menus are data in [`src/app/navigation.ts`](src/app/navigation.ts),
not markup inside components. The portal tab strip renders whatever
`PORTAL_NAV` lists, so it can grow or become a sidebar without touching pages.

### Adding a portal page

1. Create `src/pages/portal/SomethingPage.tsx`.
2. Add its path to `PATHS.portal` in `paths.ts`.
3. Add a `<Route>` inside the portal route in `routes.tsx`.
4. Add `{ label, to }` to `PORTAL_NAV` in `navigation.ts`.

Read live data with `usePipeline()`; don't call the polling hooks again.

## PipelineProvider

[`src/state/PipelineProvider.tsx`](src/state/PipelineProvider.tsx) runs
`usePipelineStatus` and `useEventFeed` **once** for everything beneath it and
exposes the result through `usePipeline()`:

```ts
const { status, feed, visible, reachable, frame, stats } = usePipeline();
```

| Field | Meaning |
|---|---|
| `status` | Last status response, error, view state, stale time, restart count |
| `feed` | Event log (newest first, max 200) and its last error |
| `visible` | False while the tab is hidden; polling and the stream pause |
| `reachable` | Backend answering (waiting, live or stale) |
| `frame`, `stats` | Current frame and pipeline stats, or `null` before the first frame |

Because it lives in `PortalLayout`, all portal pages share one status loop, one
event cursor and one restart detector, and public pages never poll.

## Shared components

| Component | Folder | Use |
|---|---|---|
| `Panel` | `ui/` | Bordered panel with a title bar; the base of every dashboard card |
| `ExternalLink` | `ui/` | New-tab link that announces it to screen readers |
| `LogoMark`, `Wordmark` | `brand/` | The logo, and logo plus name |
| `PublicNavbar`, `Footer` | `layout/` | Public site frame |
| `PortalHeader`, `PortalNav` | `layout/` | Portal status bar and tab strip |
| `StateBadge` | `pipeline/` | LIVE / STALE / OFFLINE… block |
| `LiveStream`, `StatsPanel`, `StatusPanel`, `ModulesPanel`, `FrameResults`, `EventFeed`, `ConnectionNotice` | `pipeline/` | Dashboard panels; take plain props, so any portal page can reuse them |

## Connecting to the backend

The dashboard server only starts when asked. From the **repository root**, in
the Python virtual environment:

```bash
# webcam
python -m backend.main --no-window --dashboard

# or a video file, looping at real-time speed (no webcam needed)
python -m backend.main --no-window --dashboard --video-path path/to/clip.mp4 --loop --playback-speed 1.0
```

It listens on `http://127.0.0.1:8080`. The Vite dev and preview servers proxy
`/api/*` and `/stream.mjpg` there, so the browser sees a single origin and no
CORS configuration is needed (the backend has none).

To point at a different address, set `VG_BACKEND` before starting Vite:

```bash
VG_BACKEND=http://127.0.0.1:9000 npm run dev          # bash
$env:VG_BACKEND="http://127.0.0.1:9000"; npm run dev  # PowerShell
```

Use `127.0.0.1`, not `localhost`: Node may resolve `localhost` to IPv6 `::1`,
while the backend binds IPv4 only.

### Endpoints used

| Endpoint | Method | Used for | How |
|---|---|---|---|
| `/api/status` | GET | Run info, stats, current frame results | Polled every 500 ms |
| `/api/events?since=<id>&limit=200` | GET | Event log | Polled every 1 s with a `since` cursor |
| `/stream.mjpg` | GET | Live video (MJPEG) | A plain `<img>`; one connection per tab |

Not used: `/api/snapshot.jpg`, and the backend's `/`, which serves the legacy
page. Every endpoint is GET-only and unauthenticated; none can control the
pipeline.

## Dashboard behaviour

### UI states

| State | Meaning | Detected by |
|---|---|---|
| Connecting | No answer from the backend yet | No successful `/api/status` |
| Offline | Backend unreachable | 2 consecutive failed polls (network error, timeout, HTTP error, bad payload) |
| Waiting | Backend up, no frame yet (a camera can take seconds to open) | `frame` is `{}` |
| Live | Frames are flowing | `frame.frame_id` advancing |
| Stale | Backend answers but video has stopped moving | `frame_id` unchanged for 3 s |
| Stopped | Pipeline ended | `running: false` (rarely seen; the process exits right after, which then reads as Offline) |

While offline or stopped, the last known values stay on screen, dimmed.

### Polling

- A new request is sent only after the previous one settles, so a slow backend
  never gets overlapping requests. Requests time out after 2.5 s.
- `/api/events` returns the *newest* `limit` events above the cursor, so the
  client always asks for the maximum (200) to avoid dropping part of a burst.
- Event ids restart at 1 when the backend restarts. The client detects a
  restart (`uptime_sec` going backwards, or `event_count` below the cursor) and
  resets the feed and the stream.
- Polling and the video stream stop while the tab is hidden, and when you leave
  the portal. Coming back starts fresh (briefly "Connecting").

### Video

- Bounding boxes, track ids and zone polygons are **drawn into the video by the
  backend** (`processing.show_boxes`, on by default). The frontend does not draw
  overlays, which would duplicate them.
- The stream carries no frame id, so boxes from `/api/status` cannot be matched
  exactly to a video frame. The stream is throttled (`dashboard.stream_fps`, 15
  by default) while status reflects every processed frame.
- The frontend reconnects by giving the `<img>` a new URL (`/stream.mjpg?r=N`)
  when the backend comes back, restarts, or the image reports an error.

## Module selection

`/portal/modules` lists every capability from
[`src/content/moduleCatalog.ts`](src/content/moduleCatalog.ts), grouped by category,
each with one status:

| Status | Meaning |
|---|---|
| Available | Works with the models the project uses out of the box |
| Requires setup | Code exists, but needs model files or an identity store first |
| Planned | Nothing to enable yet; labelled "on the development roadmap" or "product vision, not yet scheduled" |

A "Running now" tag appears when every backend module a capability needs is
listed in the connected pipeline's `info.modules`.

**Selection is UI-only.** The backend has no endpoint to enable modules, so the
selection lives in component state: not saved, not sent, gone on reload. The
summary panel turns it into the `modules:` block for `configs/config.yaml`
(dependencies included), which is how modules are enabled today. When a
backend activation endpoint exists, replace that panel's output with a call to
it. Update `moduleCatalog.ts` whenever a module's backend status changes.

## Public site content

The pages describe only what the backend supports today, using the repository
README's module table as the source of truth. Modules that also need model
files the project does not ship are labelled "Needs model files". The Landing
page's roadmap lists undelivered items from
[`docs/ROADMAP.md`](../../docs/ROADMAP.md) in a visibly separate, dashed
"Planned — not available yet" section. Keep both rules when editing copy.

Contact details: there is no contact form (the backend has no endpoint for
one). Add official channels to `OFFICIAL_CONTACTS` in
[`src/app/site.ts`](src/app/site.ts) and the Contact page lists them; until
then it points to the GitHub issue tracker.

## Branding

The mark is an angular eye inside the four corner brackets of a detection box,
the same brackets the backend draws around what it detects.

- `src/components/brand/Logo.tsx`: `LogoMark` (the symbol) and `Wordmark`
  (symbol plus the name in Space Grotesk). Inline SVG on a 32-unit grid.
- Colour comes from CSS: the brackets use `currentColor`, the eye uses
  `--logo-accent` (defaults to `--primary`). Pass `monochrome` for a one-colour
  version, or set `--logo-accent` on a parent to recolour the eye.
- `public/favicon.svg`: the same paths with fixed colours on a dark tile, so it
  reads on light and dark browser tabs. Update it when the paths in `Logo.tsx`
  change.
- Fonts (Space Grotesk, IBM Plex Mono) are self-hosted via Fontsource: bundled
  into `dist/`, with no request to a font CDN.
- No PNG icons yet, so iOS home-screen icons and older browsers fall back to
  their defaults.

## Implemented vs planned

| Implemented | Planned (not started) |
|---|---|
| Public Landing, About, Contact, 404 | Auth pages: login, register, password reset |
| Portal layout with header, tabs and connection notice | Portal pages: Live Monitoring, Cameras, Events & Alerts, Zones, Employees, Reports, Notifications, Profile, Settings |
| Module selection page (review, select, config snippet; not saved) | Saving or applying module selections (needs a backend endpoint) |
| Live dashboard: MJPEG video, telemetry, current frame, source, modules, event log | Anything needing new backend endpoints (authentication, camera or zone management, configuration, reports) |
| Shared polling via `PipelineProvider` | |
| Connection states, restart detection, tab-visibility pausing | |

Live Monitoring and Events & Alerts can be built from data the backend
already exposes. The other portal pages need backend support that does not
exist yet; build them as clearly labelled placeholders rather than with invented
data.

## Known limitations

- **The API lives and dies with the pipeline.** When a video file ends, the
  camera fails, or the pipeline is stopped, the backend process exits. The
  portal shows Offline until someone restarts it.
- **No schema or versioning.** `src/api/types.ts` mirrors
  `backend/services/state.py`, `computer_vision/core/context.py` and
  `computer_vision/utils/timing.py` by hand. Update it when those change.
- **Module summaries are not exposed.** Crowd count, zone occupancy,
  attendance lists and fall scores are computed by the backend but not included
  in `/api/status`. Only detections, tracks, poses and events are available.
- **No source-health or module-health fields.** An RTSP reconnect shows up only
  as Stale. A module disabled after repeated failures stays listed in
  `info.modules`; its errors appear only in the frame's `errors` while it fails.
- **Enabling `face_detection` is expected to break `/api/status`.** The module
  puts a non-JSON-serialisable object into each detection's `extra`, which should
  make the endpoint return HTTP 500 while a face is in view (found by reading the
  code, not yet reproduced). The portal would show this as Offline.
- **Production serving is not wired up.** `npm run build` produces `dist/`, but
  the backend only serves `frontend/static/index.html` at `/`. To deploy, serve
  `dist/` from a web server that forwards `/api` and `/stream.mjpg` to the
  backend and answers unknown paths with `index.html`, so links such as
  `/portal/dashboard` work when opened directly. `npm run dev` and
  `npm run preview` already do this.
- **No authentication.** Keep the backend on `127.0.0.1`. If auth is added
  later, note that an `<img>` cannot send an `Authorization` header, so the
  stream would need cookie- or query-based auth.
- **One stream per tab.** Each MJPEG connection holds one of the backend's
  worker threads, shared with the JSON endpoints.
- The event log keeps the latest 200 events in memory; there is no history
  across page reloads beyond what the backend still holds (200 by default).
