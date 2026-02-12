import React, { useState } from 'react';
import {
  AppBar,
  Box,
  Drawer,
  IconButton,
  Toolbar,
  Typography,
  useTheme,
  useMediaQuery,
  Badge,
} from '@mui/material';
import {
  Menu as MenuIcon,
  Close as CloseIcon,
  Brightness4 as DarkModeIcon,
  Brightness7 as LightModeIcon,
  Notifications as NotificationsIcon,
} from '@mui/icons-material';
import { useUIStore } from '@stores/uiStore';
import { Outlet, useLocation } from 'react-router-dom';
import Sidebar from './Sidebar';
import WebSocketStatusIndicator from './WebSocketStatusIndicator';

const DRAWER_WIDTH = 280;

interface LayoutProps {
  onThemeToggle: () => void;
}

const Layout: React.FC<LayoutProps> = ({ onThemeToggle }) => {
  const { sidebarOpen, setSidebarOpen, theme, notificationCount } = useUIStore();
  const [mobileDrawerOpen, setMobileDrawerOpen] = useState(false);
  const muiTheme = useTheme();
  const isMobile = useMediaQuery(muiTheme.breakpoints.down('sm'));
  const location = useLocation();

  // Get page title from current route
  const getPageTitle = () => {
    switch (location.pathname) {
      case '/':
        return 'Dashboard';
      case '/flows':
        return 'Network Flows';
      case '/alerts':
        return 'Security Alerts';
      case '/incidents':
        return 'Incidents';
      case '/analytics':
        return 'Analytics';
      case '/search':
        return 'Search';
      default:
        return 'AegisPCAP';
    }
  };

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh' }}>
      {/* App Bar */}
      <AppBar
        position="fixed"
        sx={{
          width: { xs: '100%', sm: `calc(100% - ${sidebarOpen && !isMobile ? DRAWER_WIDTH : 0}px)` },
          ml: { xs: 0, sm: sidebarOpen && !isMobile ? DRAWER_WIDTH : 0 },
          transition: (theme) =>
            theme.transitions.create(['width', 'margin'], {
              easing: theme.transitions.easing.sharp,
              duration: theme.transitions.duration.enteringScreen,
            }),
          backgroundColor: theme === 'dark' ? '#1e1e1e' : '#ffffff',
          color: theme === 'dark' ? '#ffffff' : '#000000',
          boxShadow: theme === 'dark' ? 'none' : '0 2px 4px rgba(0,0,0,0.1)',
          borderBottom: `1px solid ${theme === 'dark' ? '#333333' : '#eeeeee'}`,
        }}
      >
        <Toolbar>
          <IconButton
            edge="start"
            color="inherit"
            aria-label="menu"
            onClick={() => {
              if (isMobile) {
                setMobileDrawerOpen(!mobileDrawerOpen);
              } else {
                setSidebarOpen(!sidebarOpen);
              }
            }}
            sx={{ mr: 2 }}
          >
            {isMobile && mobileDrawerOpen ? <CloseIcon /> : <MenuIcon />}
          </IconButton>

          <Typography variant="h6" sx={{ flexGrow: 1, fontWeight: 600 }}>
            {getPageTitle()}
          </Typography>

          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <WebSocketStatusIndicator />

            <IconButton
              size="small"
              aria-label="notifications"
              sx={{ position: 'relative' }}
            >
              <Badge badgeContent={notificationCount} color="error">
                <NotificationsIcon />
              </Badge>
            </IconButton>

            <IconButton size="small" onClick={onThemeToggle} aria-label="toggle theme">
              {theme === 'light' ? <DarkModeIcon /> : <LightModeIcon />}
            </IconButton>
          </Box>
        </Toolbar>
      </AppBar>

      {/* Sidebar - Desktop */}
      {!isMobile && (
        <Drawer
          variant="persistent"
          open={sidebarOpen}
          sx={{
            width: DRAWER_WIDTH,
            flexShrink: 0,
            '& .MuiDrawer-paper': {
              width: DRAWER_WIDTH,
              boxSizing: 'border-box',
              mt: '64px',
              height: 'calc(100vh - 64px)',
              backgroundColor: theme === 'dark' ? '#1e1e1e' : '#fafafa',
              borderRight: `1px solid ${theme === 'dark' ? '#333333' : '#eeeeee'}`,
            },
          }}
        >
          <Sidebar onClose={() => setSidebarOpen(false)} />
        </Drawer>
      )}

      {/* Sidebar - Mobile */}
      {isMobile && (
        <Drawer
          variant="temporary"
          open={mobileDrawerOpen}
          onClose={() => setMobileDrawerOpen(false)}
          sx={{
            '& .MuiDrawer-paper': {
              width: DRAWER_WIDTH,
              boxSizing: 'border-box',
              backgroundColor: theme === 'dark' ? '#1e1e1e' : '#fafafa',
            },
          }}
        >
          <Sidebar onClose={() => setMobileDrawerOpen(false)} />
        </Drawer>
      )}

      {/* Main Content */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: 3,
          mt: '64px',
          ml: { xs: 0, sm: sidebarOpen && !isMobile ? DRAWER_WIDTH : 0 },
          transition: (theme) =>
            theme.transitions.create(['margin-left'], {
              easing: theme.transitions.easing.sharp,
              duration: theme.transitions.duration.enteringScreen,
            }),
          backgroundColor: theme === 'dark' ? '#121212' : '#f5f5f5',
          minHeight: 'calc(100vh - 64px)',
        }}
      >
        <Outlet />
      </Box>
    </Box>
  );
};

export default Layout;
