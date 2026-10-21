import { BrowserRouter } from 'react-router';
import { AssistantWidget } from '../components/assistant/AssistantWidget';
import { AppRoutes } from './routes';

export function App() {
  return (
    <BrowserRouter>
      <AppRoutes />
      {/* Outside the layouts: one instance for every page, kept across navigation. */}
      <AssistantWidget />
    </BrowserRouter>
  );
}
