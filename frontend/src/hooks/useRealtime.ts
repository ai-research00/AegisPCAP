import { useEffect, useCallback, useRef } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { useWebSocket, WebSocketChannel } from '@contexts/WebSocketContext';
import { useUIStore } from '@stores/uiStore';
import type { Alert, Incident, Flow } from '@/types/api';

/**
 * Hook to sync real-time alerts with React Query cache
 */
export const useRealtimeAlerts = () => {
  const { subscribe, unsubscribe, onAlert, isSubscribed } = useWebSocket();
  const queryClient = useQueryClient();
  const callbackRef = useRef<((data: any) => void) | null>(null);

  useEffect(() => {
    // Subscribe to alerts channel
    if (!isSubscribed(WebSocketChannel.ALERTS)) {
      subscribe(WebSocketChannel.ALERTS);
    }

    // Handle incoming alerts
    const handleAlert = (data: any) => {
      console.log('[Real-time] New alert:', data);

      // Increment notification count
      useUIStore.setState(state => ({
        notificationCount: state.notificationCount + 1,
      }));

      // Invalidate alerts query to force refetch
      queryClient.invalidateQueries({ queryKey: ['alerts'] });

      // Optionally show toast notification (handled by component)
    };

    callbackRef.current = handleAlert;
    onAlert(handleAlert);

    return () => {
      // Cleanup
      if (callbackRef.current) {
        // Note: We don't unsubscribe to keep receiving alerts
        // This allows multiple components to listen
      }
    };
  }, [subscribe, unsubscribe, onAlert, isSubscribed, queryClient]);
};

/**
 * Hook to sync real-time incidents with React Query cache
 */
export const useRealtimeIncidents = () => {
  const { subscribe, onIncident, isSubscribed } = useWebSocket();
  const queryClient = useQueryClient();
  const callbackRef = useRef<((data: any) => void) | null>(null);

  useEffect(() => {
    // Subscribe to incidents channel
    if (!isSubscribed(WebSocketChannel.INCIDENTS)) {
      subscribe(WebSocketChannel.INCIDENTS);
    }

    // Handle incoming incidents
    const handleIncident = (data: any) => {
      console.log('[Real-time] New incident:', data);

      // Increment notification count
      useUIStore.setState(state => ({
        notificationCount: state.notificationCount + 1,
      }));

      // Invalidate incidents query
      queryClient.invalidateQueries({ queryKey: ['incidents'] });
    };

    callbackRef.current = handleIncident;
    onIncident(handleIncident);

    return () => {
      // Cleanup handled by provider
    };
  }, [subscribe, onIncident, isSubscribed, queryClient]);
};

/**
 * Hook to sync real-time flow updates
 */
export const useRealtimeFlows = () => {
  const { subscribe, onFlowUpdate, isSubscribed } = useWebSocket();
  const queryClient = useQueryClient();
  const callbackRef = useRef<((data: any) => void) | null>(null);

  useEffect(() => {
    // Subscribe to flows channel
    if (!isSubscribed(WebSocketChannel.FLOWS)) {
      subscribe(WebSocketChannel.FLOWS);
    }

    // Handle flow updates
    const handleFlowUpdate = (data: any) => {
      console.log('[Real-time] Flow update:', data);

      // Invalidate flows query (soft refresh - doesn't interrupt user)
      queryClient.invalidateQueries({ 
        queryKey: ['flows'],
        exact: false,
      });
    };

    callbackRef.current = handleFlowUpdate;
    onFlowUpdate(handleFlowUpdate);

    return () => {
      // Cleanup
    };
  }, [subscribe, onFlowUpdate, isSubscribed, queryClient]);
};

/**
 * Hook to sync real-time statistics and update dashboard metrics
 */
export const useRealtimeStatistics = () => {
  const { subscribe, onStatistics, isSubscribed } = useWebSocket();
  const queryClient = useQueryClient();
  const callbackRef = useRef<((data: any) => void) | null>(null);

  useEffect(() => {
    // Subscribe to statistics channel
    if (!isSubscribed(WebSocketChannel.STATISTICS)) {
      subscribe(WebSocketChannel.STATISTICS);
    }

    // Handle statistics updates
    const handleStatistics = (data: any) => {
      console.log('[Real-time] Statistics update:', data);

      // Update dashboard metrics in cache
      queryClient.setQueryData(['dashboard', 'metrics'], data);
    };

    callbackRef.current = handleStatistics;
    onStatistics(handleStatistics);

    return () => {
      // Cleanup
    };
  }, [subscribe, onStatistics, isSubscribed, queryClient]);
};

/**
 * Hook to sync real-time network topology updates
 */
export const useRealtimeTopology = () => {
  const { subscribe, onTopologyUpdate, isSubscribed } = useWebSocket();
  const queryClient = useQueryClient();
  const callbackRef = useRef<((data: any) => void) | null>(null);

  useEffect(() => {
    // Subscribe to topology channel
    if (!isSubscribed(WebSocketChannel.TOPOLOGY)) {
      subscribe(WebSocketChannel.TOPOLOGY);
    }

    // Handle topology updates
    const handleTopologyUpdate = (data: any) => {
      console.log('[Real-time] Topology update:', data);

      // Update network graph in cache
      queryClient.setQueryData(['network', 'graph'], data);

      // Also invalidate stats to be safe
      queryClient.invalidateQueries({ queryKey: ['network', 'stats'] });
    };

    callbackRef.current = handleTopologyUpdate;
    onTopologyUpdate(handleTopologyUpdate);

    return () => {
      // Cleanup
    };
  }, [subscribe, onTopologyUpdate, isSubscribed, queryClient]);
};

/**
 * Hook for custom real-time listeners
 * Allows components to listen to specific message types
 */
export const useRealtimeListener = (
  channels: WebSocketChannel[],
  onData: (data: any) => void
) => {
  const ws = useWebSocket();

  useEffect(() => {
    // Subscribe to all requested channels
    channels.forEach(channel => {
      if (!ws.isSubscribed(channel)) {
        ws.subscribe(channel);
      }
    });

    // Register listener based on channel type
    const handlers: any[] = [];

    channels.forEach(channel => {
      let handler: (data: any) => void;

      switch (channel) {
        case WebSocketChannel.ALERTS:
          handler = (data) => onData({ type: 'alert', data });
          ws.onAlert(handler);
          handlers.push({ type: 'alert', handler });
          break;
        case WebSocketChannel.INCIDENTS:
          handler = (data) => onData({ type: 'incident', data });
          ws.onIncident(handler);
          handlers.push({ type: 'incident', handler });
          break;
        case WebSocketChannel.FLOWS:
          handler = (data) => onData({ type: 'flow', data });
          ws.onFlowUpdate(handler);
          handlers.push({ type: 'flow', handler });
          break;
        case WebSocketChannel.STATISTICS:
          handler = (data) => onData({ type: 'statistics', data });
          ws.onStatistics(handler);
          handlers.push({ type: 'statistics', handler });
          break;
        case WebSocketChannel.TOPOLOGY:
          handler = (data) => onData({ type: 'topology', data });
          ws.onTopologyUpdate(handler);
          handlers.push({ type: 'topology', handler });
          break;
      }
    });

    return () => {
      // Note: In production, you might want to track cleanup
      // For now, we keep listeners active across component remounts
    };
  }, [channels, onData, ws]);
};

/**
 * Hook to show toast notification on new alerts
 */
export const useAlertNotification = () => {
  const { subscribe, onAlert, isSubscribed } = useWebSocket();
  const showToast = useCallback((message: string, type?: 'success' | 'error' | 'warning' | 'info') => {
    // This would integrate with your toast system
    console.log(`[Toast] ${type}: ${message}`);
  }, []);

  useEffect(() => {
    if (!isSubscribed(WebSocketChannel.ALERTS)) {
      subscribe(WebSocketChannel.ALERTS);
    }

    const handleAlert = (data: any) => {
      const title = data.title || 'New Alert';
      const severity = data.severity || 'info';
      
      // Map severity to toast type
      const toastType = 
        severity === 'critical' || severity === 'high' ? 'error' :
        severity === 'medium' ? 'warning' :
        'info';

      showToast(title, toastType);
    };

    onAlert(handleAlert);
  }, [subscribe, onAlert, isSubscribed, showToast]);
};
