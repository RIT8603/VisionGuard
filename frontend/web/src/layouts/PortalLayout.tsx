import { Outlet } from 'react-router';
import { PORTAL_NAV } from '../app/navigation';
import { PortalHeader } from '../components/layout/PortalHeader';
import { PortalNav } from '../components/layout/PortalNav';
import { ConnectionNotice } from '../components/pipeline/ConnectionNotice';
import { PipelineProvider, usePipeline } from '../state/PipelineProvider';
import { BackendGate } from './BackendGate';

/**
 * Frame for every portal page: status header, tab navigation, connection
 * notice, then the page. Owns the PipelineProvider, so polling runs only
 * while a portal page is open and is shared by all of them. BackendGate keeps
 * it closed (showing the startup screen) until the GPU backend is ready.
 */
export function PortalLayout() {
  return (
    <BackendGate>
      <PipelineProvider>
        <PortalShell />
      </PipelineProvider>
    </BackendGate>
  );
}

function PortalShell() {
  const { status, cameras } = usePipeline();
  const { data } = status;

  return (
    <div className="app">
      <a className="skip-link" href="#portal-main">
        Skip to content
      </a>
      <PortalHeader
        info={data?.info}
        viewState={status.viewState}
        uptimeSec={data?.uptime_sec}
        cameras={cameras.data?.cameras}
      />
      <PortalNav items={PORTAL_NAV} />
      <ConnectionNotice status={status} />
      <main id="portal-main">
        <Outlet />
      </main>
    </div>
  );
}
