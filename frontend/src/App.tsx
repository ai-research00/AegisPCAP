import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClientProvider } from '@tanstack/react-query';
import { queryClient } from '@utils/queryClient';
import { ThemeProvider } from '@theme/index';
import { useUIStore } from '@stores/uiStore';
import { ErrorBoundary } from '@components/ErrorBoundary';
import { WebSocketProvider } from '@contexts/WebSocketContext';
import Layout from '@components/Layout';
import DashboardPage from '@pages/DashboardPage';
import FlowsPage from '@pages/FlowsPage';
import AlertsPage from '@pages/AlertsPage';
import IncidentsPage from '@pages/IncidentsPage';

// Lazy load other pages
const NetworkVisualizationPage = React.lazy(() => import('@pages/NetworkVisualizationPage'));
const AnalyticsPage = React.lazy(() => import('@pages/AnalyticsPage'));
const SearchPage = React.lazy(() => import('@pages/SearchPage'));

function App() {
  const { theme, toggleTheme } = useUIStore();

  return (
    <ErrorBoundary>
      <QueryClientProvider client={queryClient}>
        <WebSocketProvider autoConnect={true} reconnectAttempts={5}>
          <ThemeProvider theme={theme}>
            <Router>
              <Routes>
              <Route path="/" element={<Layout onThemeToggle={toggleTheme} />}>
                <Route
                  index
                  element={
                    <ErrorBoundary>
                      <DashboardPage />
                    </ErrorBoundary>
                  }
                />
                <Route
                  path="flows"
                  element={
                    <ErrorBoundary>
                      <FlowsPage />
                    </ErrorBoundary>
                  }
                />
                <Route
                  path="alerts"
                  element={
                    <ErrorBoundary>
                      <AlertsPage />
                    </ErrorBoundary>
                  }
                />
                <Route
                  path="incidents"
                  element={
                    <ErrorBoundary>
                      <IncidentsPage />
                    </ErrorBoundary>
                  }
                />
                <Route
                  path="network"
                  element={
                    <ErrorBoundary>
                      <React.Suspense fallback={<div>Loading...</div>}>
                        <NetworkVisualizationPage />
                      </React.Suspense>
                    </ErrorBoundary>
                  }
                />
                <Route
                  path="analytics"
                  element={
                    <ErrorBoundary>
                      <React.Suspense fallback={<div>Loading...</div>}>
                        <AnalyticsPage />
                      </React.Suspense>
                    </ErrorBoundary>
                  }
                />
                <Route
                  path="search"
                  element={
                    <ErrorBoundary>
                      <React.Suspense fallback={<div>Loading...</div>}>
                        <SearchPage />
                      </React.Suspense>
                    </ErrorBoundary>
                  }
                />
              </Route>
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </Router>
        </ThemeProvider>
        </WebSocketProvider>
      </QueryClientProvider>
    </ErrorBoundary>
  );
}

export default App;
