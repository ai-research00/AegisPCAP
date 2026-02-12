import React from 'react';
import {
  Box,
  Chip,
  Tooltip,
  CircularProgress,
} from '@mui/material';
import {
  FiberManualRecord as CircleIcon,
  CloudOff as DisconnectedIcon,
  CloudDone as ConnectedIcon,
  Warning as WarningIcon,
} from '@mui/icons-material';
import { useWebSocket, ConnectionStatus } from '@contexts/WebSocketContext';

/**
 * WebSocket Connection Status Indicator
 * Shows real-time connection status in the UI
 */
const WebSocketStatusIndicator: React.FC = () => {
  const { status, error } = useWebSocket();

  const getStatusColor = (): string => {
    switch (status) {
      case ConnectionStatus.CONNECTED:
        return '#4caf50'; // Green
      case ConnectionStatus.CONNECTING:
      case ConnectionStatus.RECONNECTING:
        return '#ff9800'; // Orange
      case ConnectionStatus.DISCONNECTED:
      case ConnectionStatus.ERROR:
        return '#f44336'; // Red
      default:
        return '#9e9e9e'; // Gray
    }
  };

  const getStatusLabel = (): string => {
    switch (status) {
      case ConnectionStatus.CONNECTED:
        return 'Real-time Connected';
      case ConnectionStatus.CONNECTING:
        return 'Connecting...';
      case ConnectionStatus.RECONNECTING:
        return 'Reconnecting...';
      case ConnectionStatus.DISCONNECTED:
        return 'Disconnected';
      case ConnectionStatus.ERROR:
        return 'Connection Error';
      default:
        return 'Unknown';
    }
  };

  const getIcon = () => {
    switch (status) {
      case ConnectionStatus.CONNECTED:
        return <ConnectedIcon sx={{ fontSize: 18 }} />;
      case ConnectionStatus.CONNECTING:
      case ConnectionStatus.RECONNECTING:
        return <CircularProgress size={18} sx={{ color: 'white' }} />;
      case ConnectionStatus.DISCONNECTED:
        return <DisconnectedIcon sx={{ fontSize: 18 }} />;
      case ConnectionStatus.ERROR:
        return <WarningIcon sx={{ fontSize: 18 }} />;
      default:
        return <CircleIcon sx={{ fontSize: 12 }} />;
    }
  };

  const tooltipTitle = error ? `${getStatusLabel()}: ${error}` : getStatusLabel();

  return (
    <Tooltip title={tooltipTitle}>
      <Chip
        icon={getIcon()}
        label={getStatusLabel()}
        size="small"
        sx={{
          backgroundColor: getStatusColor(),
          color: 'white',
          fontWeight: 500,
          cursor: 'pointer',
          '& .MuiChip-icon': {
            color: 'white !important',
          },
        }}
      />
    </Tooltip>
  );
};

export default WebSocketStatusIndicator;
