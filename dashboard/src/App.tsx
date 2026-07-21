import { useSignal } from './hooks/useSignal';
import { MobileSignalBar } from './components/MobileSignalBar';
import { DesktopDashboard } from './components/DesktopDashboard';
import { useEffect, useState } from 'react';

function App() {
  const { prediction, history, connected } = useSignal();
  const [isMobile, setIsMobile] = useState(false);

  useEffect(() => {
    const check = () => {
      setIsMobile(window.innerWidth < 1024);
    };
    check();
    window.addEventListener('resize', check);
    return () => window.removeEventListener('resize', check);
  }, []);

  return isMobile ? (
    <MobileSignalBar prediction={prediction} history={history} connected={connected} />
  ) : (
    <DesktopDashboard prediction={prediction} history={history} connected={connected} />
  );
}

export default App;
