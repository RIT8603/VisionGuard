# VisionGuard web frontend

React + Vite + TypeScript dashboard for a running VisionGuard pipeline: live
video, run status, per-frame detections and tracks, the event feed, and
performance stats.

It is a read-only client of the existing dashboard API in
[`backend/api/server.py`](../../backend/api/server.py). It does not change the
backend, the vision pipeline, or the legacy page in
[`frontend/static/`](../static/), which keeps working as before.

## Requirements

- Node.js 20 or newer (npm comes with it)
- The VisionGuard backend running with its dashboard enabled (see below)

## Setup

```bash
cd frontend/web
npm install
npm run dev
```

Open <http://localhost:5173>.

| Script | What it does |
|---|---|
| `npm run dev` | Vite dev server with the backend proxy |
| `npm run typecheck` | TypeScript only, no output |
| `npm run build` | Type-check, then production build into `dist/` |
| `npm run preview` | Serve `dist/` locally, with the same proxy |

## Connecting to the backend

The dashboard server only starts when asked. From the **repository root**, in
the Python virtual environment:

```bash
# webcam
python -m backend.main --no-window --dashboard

# or a video file, looping at real-time speed (no webcam needed)
python -m backend.main --no-window --dashboard --video-path path/to/clip.mp4 --loop --playback-speed 1.0
```

It listens on `http://127.0.0.1:8080`. The Vite dev server proxies `/api/*` and
`/stream.mjpg` there, so the browser sees a single origin and no CORS
configuration is needed (the backend has none).

To point at a different address, set `VG_BACKEND` before starting Vite:

```bash
VG_BACKEND=http://127.0.0.1:9000 npm run dev          # bash
$env:VG_BACKEND="http://127.0.0.1:9000"; npm run dev  # PowerShell
```

Use `127.0.0.1`, not `localhost`: Node may resolve `localhost` to IPv6 `::1`,
while the backend binds IPv4 only.

## Endpoints used

| Endpoint | Method | Used for | How |
|---|---|---|---|
| `/api/status` | GET | Run info, stats, current frame results | Polled every 500 ms |
| `/api/events?since=<id>&limit=200` | GET | Event feed | Polled every 1 s with a `since` cursor |
| `/stream.mjpg` | GET | Live video (MJPEG) | A plain `<img>`; one connection per tab |

Not used: `/api/snapshot.jpg`, and `/`, which serves the legacy page.

The backend has no authentication. Every endpoint is GET-only; it cannot be
used to control the pipeline.

## How it works

```
src/
  config.ts              poll intervals, timeouts, thresholds
  api/
    types.ts             response models (hand-written; see "Known limitations")
    guards.ts            runtime shape checks
    client.ts            fetch wrapper: timeout, cancellation, typed ApiError
    endpoints.ts         getStatus, getEvents, streamUrl
  state/viewState.ts     derives the UI state from the status poll
  hooks/
    usePipelineStatus.ts status polling, stale and restart detection
    useEventFeed.ts      event polling with a since cursor
    usePageVisible.ts    pauses everything while the tab is hidden
  components/            layout panels
  lib/format.ts          number, time and event formatting
```

### UI states

| State | Meaning | Detected by |
|---|---|---|
| Connecting | No answer from the backend yet | No successful `/api/status` |
| Offline | Backend unreachable | 2 consecutive failed polls (network error, timeout, HTTP error, bad payload) |
| Waiting for video | Backend up, no frame yet (a camera can take seconds to open) | `frame` is `{}` |
| Live | Frames are flowing | `frame.frame_id` advancing |
| Stale | Backend answers but video has stopped moving | `frame_id` unchanged for 3 s |
| Stopped | Pipeline ended | `running: false` (rarely seen; the process exits right after, which then reads as Offline) |

While offline or stopped, the last known values stay on screen, dimmed.

### Polling details

- A new request is sent only after the previous one settles, so a slow backend
  never gets overlapping requests. Requests time out after 2.5 s.
- `/api/events` returns the *newest* `limit` events above the cursor, so the
  client always asks for the maximum (200) to avoid dropping part of a burst.
- Event ids restart at 1 when the backend restarts. The client detects a
  restart (`uptime_sec` going backwards, or `event_count` below the cursor) and
  resets the feed and the stream.
- Polling and the video stream stop while the tab is hidden and resume when it
  is visible again.

### Video

- Bounding boxes, track ids and zone polygons are **drawn into the video by the
  backend** (`processing.show_boxes`, on by default). The frontend does not draw
  overlays, which would duplicate them.
- The stream carries no frame id, so boxes from `/api/status` cannot be matched
  exactly to a video frame. The stream is also throttled (`dashboard.stream_fps`,
  15 by default) while status reflects every processed frame.
- The frontend reconnects by giving the `<img>` a new URL (`/stream.mjpg?r=N`)
  when the backend comes back, restarts, or the image reports an error.

## Branding

The mark is an angular eye inside the four corner brackets of a detection box,
the same brackets the backend draws around what it detects.

- `src/components/Logo.tsx`: `LogoMark` (the symbol) and `Wordmark` (symbol plus
  the name in Space Grotesk, used in the header). Inline SVG on a 32-unit grid.
- Colour comes from CSS: the brackets use `currentColor`, the eye uses
  `--logo-accent` (defaults to `--primary`). Pass `monochrome` for a one-colour
  version, or set `--logo-accent` on a parent to recolour the eye.
- `public/favicon.svg`: the same paths with fixed colours on a dark tile, so it
  reads on light and dark browser tabs. Update it when the paths in `Logo.tsx`
  change.
- No PNG icons yet, so iOS home-screen icons and older browsers fall back to
  their defaults.

## Known limitations

These come from the current backend, which is treated as stable for this
version:

- **The API lives and dies with the pipeline.** When a video file ends, the
  camera fails, or the pipeline is stopped, the backend process exits. The UI
  shows Offline until someone restarts it.
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
  code, not yet reproduced). The UI would show this as Offline.
- **Production serving is not wired up.** `npm run build` produces `dist/`, but
  the backend only serves `frontend/static/index.html` at `/`. To deploy, serve
  `dist/` from a web server that also forwards `/api` and `/stream.mjpg` to the
  backend.
- **No authentication.** Keep the backend on `127.0.0.1`. If auth is added
  later, note that an `<img>` cannot send an `Authorization` header, so the
  stream would need cookie- or query-based auth.
- **One stream per tab.** Each MJPEG connection holds one of the backend's
  worker threads, shared with the JSON endpoints.
- The event feed keeps the latest 200 events in memory; there is no history
  across page reloads beyond what the backend still holds (200 by default).
