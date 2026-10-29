import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react';
import { getWakeStatus, requestWake, type WakeStatus } from '../api/wake';
import { StartupScreen } from '../components/wake/StartupScreen';
import { usePageVisible } from '../hooks/usePageVisible';

const STARTING_POLL_MS = 2000;
const OPEN_POLL_MS = 15000;
/** Re-send the start request this often while the backend still reports asleep. */
const RESEND_START_MS = 30000;

type Phase = 'checking' | 'starting' | 'open';

/**
 * Keeps backend-dependent pages closed until the wake service has verified the
 * API, the YOLO model and the tunnel. Opening the portal is the request for the
 * backend: a sleeping GPU server is started automatically.
 *
 * Without a wake service (local development) the portal opens directly.
 */
export function BackendGate({ children }: { children: ReactNode }) {
  const visible = usePageVisible();
  const [phase, setPhase] = useState<Phase>('checking');
  const [status, setStatus] = useState<WakeStatus | null>(null);
  const lastStartAt = useRef(0);
  const notReadyCount = useRef(0);

  const start = useCallback(async () => {
    lastStartAt.current = Date.now();
    const reply = await requestWake();
    if (reply) setStatus(reply);
  }, []);

  // First look: open directly, or start the backend.
  useEffect(() => {
    let cancelled = false;
    getWakeStatus().then((reply) => {
      if (cancelled) return;
      if (reply === null || reply.ready) {
        setPhase('open');
        if (reply) setStatus(reply);
        return;
      }
      setStatus(reply);
      setPhase('starting');
      void start();
    });
    return () => {
      cancelled = true;
    };
  }, [start]);

  // Poll while starting (fast) or while open (slow, to notice an idle stop).
  useEffect(() => {
    if (phase === 'checking' || !visible) return;
    let cancelled = false;
    const poll = async () => {
      const reply = await getWakeStatus();
      if (cancelled || reply === null) return;
      setStatus(reply);
      if (phase === 'starting') {
        if (reply.ready) {
          notReadyCount.current = 0;
          setPhase('open');
        } else if (
          !reply.error &&
          (reply.state === 'asleep' || reply.state === 'unknown') &&
          Date.now() - lastStartAt.current > RESEND_START_MS
        ) {
          void start();
        }
      } else if (!reply.ready) {
        // Two misses in a row, so one slow check does not close the dashboard.
        notReadyCount.current += 1;
        if (notReadyCount.current >= 2) {
          setPhase('starting');
          void start();
        }
      } else {
        notReadyCount.current = 0;
      }
    };
    void poll();
    const timer = window.setInterval(poll, phase === 'starting' ? STARTING_POLL_MS : OPEN_POLL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [phase, visible, start]);

  if (phase === 'open') return <>{children}</>;
  return <StartupScreen status={status} onRetry={() => void start()} />;
}
