import React, { createContext, useContext, useState, useEffect, useRef, useCallback } from 'react';
import type { FC, ReactNode } from 'react';

/**
 * WebSocket Message Types matching backend
 */
export enum WebSocketMessageType {
  ALERT = 'alert',
  INCIDENT = 'incident',
  FLOW_UPDATE = 'flow_update',
  STATISTICS = 'statistics',
  TOPOLOGY_UPDATE = 'topology_update',
  HEARTBEAT = 'heartbeat',
  SUBSCRIBE = 'subscribe',
  UNSUBSCRIBE = 'unsubscribe',
  CONNECTION_ESTABLISHED = 'connection_established',
  SUBSCRIBED = 'subscribed',
  UNSUBSCRIBED = 'unsubscribed',
}

/**
 * Generic WebSocket Message
 */
export interface WebSocketMessage<T = any> {
  type: WebSocketMessageType;
  timestamp?: string;
  data: T;
  channel?: string;
}

/**
 * WebSocket Channel Types
 */
export enum WebSocketChannel {
  ALERTS = 'alerts',
  INCIDENTS = 'incidents',
  FLOWS = 'flows',
  TOPOLOGY = 'topology',
  STATISTICS = 'statistics',
}

/**
 * WebSocket Connection Status
 */
export enum ConnectionStatus {
  DISCONNECTED = 'disconnected',
  CONNECTING = 'connecting',
  CONNECTED = 'connected',
  RECONNECTING = 'reconnecting',
  ERROR = 'error',
}

/**
 * Real-time update handlers
 */
export interface WebSocketContextType {
  // Connection state
  status: ConnectionStatus;
  isConnected: boolean;
  error: string | null;

  // Subscribe/unsubscribe to channels
  subscribe: (channel: WebSocketChannel) => void;
  unsubscribe: (channel: WebSocketChannel) => void;
  isSubscribed: (channel: WebSocketChannel) => boolean;

  // Message handlers - clients register to receive updates
  onMessage: (callback: (message: WebSocketMessage) => void) => void;
  offMessage: (callback: (message: WebSocketMessage) => void) => void;

  // Channel-specific listeners
  onAlert: (callback: (data: any) => void) => void;
  onIncident: (callback: (data: any) => void) => void;
  onFlowUpdate: (callback: (data: any) => void) => void;
  onStatistics: (callback: (data: any) => void) => void;
  onTopologyUpdate: (callback: (data: any) => void) => void;

  // Manual send (for ping/keep-alive)
  send: (message: any) => void;

  // Reconnection control
  reconnect: () => void;
  disconnect: () => void;
}

/**
 * WebSocket Context
 */
const WebSocketContext = createContext<WebSocketContextType | undefined>(undefined);

/**
 * Hook to use WebSocket context
 */
export const useWebSocket = (): WebSocketContextType => {
  const context = useContext(WebSocketContext);
  if (!context) {
    throw new Error('useWebSocket must be used within WebSocketProvider');
  }
  return context;
};

/**
 * Props for WebSocketProvider
 */
interface WebSocketProviderProps {
  children: ReactNode;
  autoConnect?: boolean;
  reconnectAttempts?: number;
  reconnectDelay?: number;
}

/**
 * WebSocket Provider Component
 * Manages WebSocket connection and message broadcasting
 */
export const WebSocketProvider: FC<WebSocketProviderProps> = ({
  children,
  autoConnect = true,
  reconnectAttempts = 5,
  reconnectDelay = 3000,
}) => {
  // Connection state
  const [status, setStatus] = useState<ConnectionStatus>(ConnectionStatus.DISCONNECTED);
  const [error, setError] = useState<string | null>(null);
  const [subscribedChannels, setSubscribedChannels] = useState<Set<WebSocketChannel>>(new Set());

  // WebSocket reference
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const heartbeatIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Message handlers
  const messageHandlersRef = useRef<Set<(message: WebSocketMessage) => void>>(new Set());
  const alertHandlersRef = useRef<Set<(data: any) => void>>(new Set());
  const incidentHandlersRef = useRef<Set<(data: any) => void>>(new Set());
  const flowUpdateHandlersRef = useRef<Set<(data: any) => void>>(new Set());
  const statisticsHandlersRef = useRef<Set<(data: any) => void>>(new Set());
  const topologyUpdateHandlersRef = useRef<Set<(data: any) => void>>(new Set());

  /**
   * Get WebSocket URL
   */
  const getWebSocketUrl = useCallback((): string => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    return `${protocol}//${host}/api/dashboard/ws`;
  }, []);

  /**
   * Connect to WebSocket
   */
  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return; // Already connected
    }

    setStatus(ConnectionStatus.CONNECTING);
    setError(null);

    try {
      const wsUrl = getWebSocketUrl();
      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        console.log('[WebSocket] Connected to', wsUrl);
        setStatus(ConnectionStatus.CONNECTED);
        setError(null);
        reconnectAttemptsRef.current = 0;

        // Start heartbeat
        startHeartbeat();

        // Re-subscribe to previous channels
        subscribedChannels.forEach(channel => {
          send({
            type: WebSocketMessageType.SUBSCRIBE,
            channel,
          });
        });
      };

      ws.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);

          // Skip heartbeat messages
          if (message.type === WebSocketMessageType.HEARTBEAT) {
            return;
          }

          console.log('[WebSocket] Message:', message.type, message);

          // Broadcast to all listeners
          messageHandlersRef.current.forEach(handler => handler(message));

          // Broadcast to channel-specific listeners
          switch (message.type) {
            case WebSocketMessageType.ALERT:
              alertHandlersRef.current.forEach(handler => handler(message.data));
              break;
            case WebSocketMessageType.INCIDENT:
              incidentHandlersRef.current.forEach(handler => handler(message.data));
              break;
            case WebSocketMessageType.FLOW_UPDATE:
              flowUpdateHandlersRef.current.forEach(handler => handler(message.data));
              break;
            case WebSocketMessageType.STATISTICS:
              statisticsHandlersRef.current.forEach(handler => handler(message.data));
              break;
            case WebSocketMessageType.TOPOLOGY_UPDATE:
              topologyUpdateHandlersRef.current.forEach(handler => handler(message.data));
              break;
          }
        } catch (e) {
          console.error('[WebSocket] Error parsing message:', e);
        }
      };

      ws.onerror = (event) => {
        console.error('[WebSocket] Error:', event);
        setStatus(ConnectionStatus.ERROR);
        setError('WebSocket connection error');
      };

      ws.onclose = () => {
        console.log('[WebSocket] Disconnected');
        stopHeartbeat();

        if (reconnectAttemptsRef.current < reconnectAttempts) {
          setStatus(ConnectionStatus.RECONNECTING);
          reconnectAttemptsRef.current += 1;
          const delay = reconnectDelay * Math.pow(2, reconnectAttemptsRef.current - 1);
          console.log(`[WebSocket] Reconnecting in ${delay}ms (attempt ${reconnectAttemptsRef.current}/${reconnectAttempts})`);
          reconnectTimeoutRef.current = setTimeout(connect, delay);
        } else {
          setStatus(ConnectionStatus.DISCONNECTED);
          setError('Failed to reconnect after multiple attempts');
        }
      };

      wsRef.current = ws;
    } catch (e) {
      console.error('[WebSocket] Connection error:', e);
      setStatus(ConnectionStatus.ERROR);
      setError(e instanceof Error ? e.message : 'Unknown error');
    }
  }, [getWebSocketUrl, reconnectAttempts, reconnectDelay, subscribedChannels]);

  /**
   * Disconnect from WebSocket
   */
  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    stopHeartbeat();
    wsRef.current?.close();
    wsRef.current = null;
    setStatus(ConnectionStatus.DISCONNECTED);
    setSubscribedChannels(new Set());
  }, []);

  /**
   * Send message to server
   */
  const send = useCallback((message: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    } else {
      console.warn('[WebSocket] Not connected, cannot send message:', message);
    }
  }, []);

  /**
   * Start heartbeat to keep connection alive
   */
  const startHeartbeat = useCallback(() => {
    stopHeartbeat();
    heartbeatIntervalRef.current = setInterval(() => {
      send({ type: 'ping' });
    }, 30000); // Every 30 seconds
  }, [send]);

  /**
   * Stop heartbeat
   */
  const stopHeartbeat = useCallback(() => {
    if (heartbeatIntervalRef.current) {
      clearInterval(heartbeatIntervalRef.current);
      heartbeatIntervalRef.current = null;
    }
  }, []);

  /**
   * Subscribe to a channel
   */
  const subscribe = useCallback((channel: WebSocketChannel) => {
    setSubscribedChannels(prev => {
      const updated = new Set(prev);
      updated.add(channel);
      return updated;
    });

    send({
      type: WebSocketMessageType.SUBSCRIBE,
      channel,
    });
  }, [send]);

  /**
   * Unsubscribe from a channel
   */
  const unsubscribe = useCallback((channel: WebSocketChannel) => {
    setSubscribedChannels(prev => {
      const updated = new Set(prev);
      updated.delete(channel);
      return updated;
    });

    send({
      type: WebSocketMessageType.UNSUBSCRIBE,
      channel,
    });
  }, [send]);

  /**
   * Check if subscribed to channel
   */
  const isSubscribed = useCallback((channel: WebSocketChannel): boolean => {
    return subscribedChannels.has(channel);
  }, [subscribedChannels]);

  /**
   * Register message handler
   */
  const onMessage = useCallback((callback: (message: WebSocketMessage) => void) => {
    messageHandlersRef.current.add(callback);
  }, []);

  /**
   * Unregister message handler
   */
  const offMessage = useCallback((callback: (message: WebSocketMessage) => void) => {
    messageHandlersRef.current.delete(callback);
  }, []);

  /**
   * Channel-specific listeners
   */
  const onAlert = useCallback((callback: (data: any) => void) => {
    alertHandlersRef.current.add(callback);
  }, []);

  const onIncident = useCallback((callback: (data: any) => void) => {
    incidentHandlersRef.current.add(callback);
  }, []);

  const onFlowUpdate = useCallback((callback: (data: any) => void) => {
    flowUpdateHandlersRef.current.add(callback);
  }, []);

  const onStatistics = useCallback((callback: (data: any) => void) => {
    statisticsHandlersRef.current.add(callback);
  }, []);

  const onTopologyUpdate = useCallback((callback: (data: any) => void) => {
    topologyUpdateHandlersRef.current.add(callback);
  }, []);

  /**
   * Auto-connect on mount
   */
  useEffect(() => {
    if (autoConnect) {
      connect();
    }

    return () => {
      disconnect();
    };
  }, [autoConnect, connect, disconnect]);

  const value: WebSocketContextType = {
    status,
    isConnected: status === ConnectionStatus.CONNECTED,
    error,
    subscribe,
    unsubscribe,
    isSubscribed,
    onMessage,
    offMessage,
    onAlert,
    onIncident,
    onFlowUpdate,
    onStatistics,
    onTopologyUpdate,
    send,
    reconnect: connect,
    disconnect,
  };

  return (
    <WebSocketContext.Provider value={value}>
      {children}
    </WebSocketContext.Provider>
  );
};
