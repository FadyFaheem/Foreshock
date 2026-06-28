import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider, CssBaseline } from '@mui/material';
import AppLayout from './components/AppLayout';
import AnalyzePage from './pages/AnalyzePage';
import DiagnosticsPage from './pages/DiagnosticsPage';
import FaultLabPage from './pages/FaultLabPage';
import HealthPage from './pages/HealthPage';
import LivePage from './pages/LivePage';
import { appTheme } from './theme/theme';

function App() {
  return (
    <ThemeProvider theme={appTheme}>
      <CssBaseline />
      <BrowserRouter>
        <Routes>
          <Route element={<AppLayout />}>
            <Route index element={<AnalyzePage />} />
            <Route path="diagnostics" element={<DiagnosticsPage />} />
            <Route path="health" element={<HealthPage />} />
            <Route path="live" element={<LivePage />} />
            <Route path="fault-lab" element={<FaultLabPage />} />
          </Route>
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
    </ThemeProvider>
  );
}

export default App;
