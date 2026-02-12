import React from 'react';
import {
  Box,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Divider,
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  Memory as FlowsIcon,
  Warning as AlertsIcon,
  Flag as IncidentsIcon,
  Hub as NetworkIcon,
  AnalyticsOutlined as AnalyticsIcon,
  Search as SearchIcon,
  Settings as SettingsIcon,
} from '@mui/icons-material';
import { Link, useLocation } from 'react-router-dom';
import { useUIStore } from '@stores/uiStore';

interface SidebarProps {
  onClose?: () => void;
}

const Sidebar: React.FC<SidebarProps> = ({ onClose }) => {
  const location = useLocation();
  const { theme } = useUIStore();

  const menuItems = [
    { label: 'Dashboard', path: '/', icon: DashboardIcon },
    { label: 'Network Flows', path: '/flows', icon: FlowsIcon },
    { label: 'Security Alerts', path: '/alerts', icon: AlertsIcon },
    { label: 'Incidents', path: '/incidents', icon: IncidentsIcon },
    { label: 'Network Topology', path: '/network', icon: NetworkIcon },
    { label: 'Analytics', path: '/analytics', icon: AnalyticsIcon },
  ];

  const secondaryItems = [
    { label: 'Search', path: '/search', icon: SearchIcon },
    { label: 'Settings', path: '/settings', icon: SettingsIcon },
  ];

  const isActive = (path: string) => location.pathname === path;

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <Box sx={{ p: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
          <Box
            sx={{
              width: 32,
              height: 32,
              borderRadius: '50%',
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: 'white',
              fontWeight: 'bold',
            }}
          >
            A
          </Box>
          <Box>
            <Box sx={{ fontSize: '0.875rem', fontWeight: 600 }}>AegisPCAP</Box>
            <Box sx={{ fontSize: '0.75rem', opacity: 0.6 }}>v0.1.0</Box>
          </Box>
        </Box>
      </Box>

      <Divider />

      <List sx={{ flex: 1, py: 1 }}>
        {menuItems.map((item) => (
          <ListItem key={item.path} disablePadding>
            <ListItemButton
              component={Link}
              to={item.path}
              selected={isActive(item.path)}
              onClick={onClose}
              sx={{
                py: 1.5,
                px: 2,
                backgroundColor: isActive(item.path)
                  ? theme === 'dark'
                    ? 'rgba(144, 202, 249, 0.12)'
                    : 'rgba(25, 118, 210, 0.08)'
                  : 'transparent',
                color: isActive(item.path)
                  ? theme === 'dark'
                    ? '#90caf9'
                    : '#1976d2'
                  : 'inherit',
                '&:hover': {
                  backgroundColor:
                    theme === 'dark'
                      ? 'rgba(144, 202, 249, 0.08)'
                      : 'rgba(25, 118, 210, 0.04)',
                },
                borderRight: isActive(item.path)
                  ? `3px solid ${theme === 'dark' ? '#90caf9' : '#1976d2'}`
                  : 'none',
                transition: 'all 0.2s ease',
              }}
            >
              <ListItemIcon
                sx={{
                  minWidth: 40,
                  color: isActive(item.path) ? 'inherit' : 'default',
                }}
              >
                <item.icon />
              </ListItemIcon>
              <ListItemText primary={item.label} />
            </ListItemButton>
          </ListItem>
        ))}
      </List>

      <Divider />

      <List sx={{ py: 1 }}>
        {secondaryItems.map((item) => (
          <ListItem key={item.path} disablePadding>
            <ListItemButton
              component={Link}
              to={item.path}
              onClick={onClose}
              sx={{
                py: 1.5,
                px: 2,
                opacity: 0.7,
                '&:hover': {
                  opacity: 1,
                },
              }}
            >
              <ListItemIcon sx={{ minWidth: 40 }}>
                <item.icon />
              </ListItemIcon>
              <ListItemText primary={item.label} />
            </ListItemButton>
          </ListItem>
        ))}
      </List>
    </Box>
  );
};

export default Sidebar;
