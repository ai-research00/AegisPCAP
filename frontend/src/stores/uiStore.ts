import { create } from 'zustand';

interface UIState {
  sidebarOpen: boolean;
  theme: 'light' | 'dark';
  selectedFlowId: string | null;
  selectedAlertId: string | null;
  selectedIncidentId: string | null;
  notificationCount: number;
  
  setSidebarOpen: (open: boolean) => void;
  toggleSidebar: () => void;
  
  setTheme: (theme: 'light' | 'dark') => void;
  toggleTheme: () => void;
  
  setSelectedFlowId: (id: string | null) => void;
  setSelectedAlertId: (id: string | null) => void;
  setSelectedIncidentId: (id: string | null) => void;
  
  setNotificationCount: (count: number) => void;
  incrementNotifications: () => void;
  clearNotifications: () => void;
}

export const useUIStore = create<UIState>((set) => ({
  sidebarOpen: true,
  theme: 'light',
  selectedFlowId: null,
  selectedAlertId: null,
  selectedIncidentId: null,
  notificationCount: 0,

  setSidebarOpen: (open) =>
    set(() => ({
      sidebarOpen: open,
    })),
  toggleSidebar: () =>
    set((state) => ({
      sidebarOpen: !state.sidebarOpen,
    })),

  setTheme: (theme) =>
    set(() => ({
      theme,
    })),
  toggleTheme: () =>
    set((state) => ({
      theme: state.theme === 'light' ? 'dark' : 'light',
    })),

  setSelectedFlowId: (id) =>
    set(() => ({
      selectedFlowId: id,
    })),
  setSelectedAlertId: (id) =>
    set(() => ({
      selectedAlertId: id,
    })),
  setSelectedIncidentId: (id) =>
    set(() => ({
      selectedIncidentId: id,
    })),

  setNotificationCount: (count) =>
    set(() => ({
      notificationCount: count,
    })),
  incrementNotifications: () =>
    set((state) => ({
      notificationCount: state.notificationCount + 1,
    })),
  clearNotifications: () =>
    set(() => ({
      notificationCount: 0,
    })),
}));
