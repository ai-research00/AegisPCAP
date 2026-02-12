import { useQuery, UseQueryOptions } from '@tanstack/react-query';
import apiClient from '@services/apiClient';
import type { PaginatedResponse, Flow, Alert, Incident } from '@/types/api';

// Flow hooks
export const useFlows = (filters: any) => {
  return useQuery({
    queryKey: ['flows', filters],
    queryFn: async () => {
      const response = await apiClient.get<PaginatedResponse<Flow>>(
        `/flows?page=${filters.page || 0}&page_size=${filters.pageSize || 50}`
      );
      return response;
    },
  });
};

export const useFlowDetails = (flowId: string) => {
  return useQuery({
    queryKey: ['flows', flowId],
    queryFn: async () => {
      const response = await apiClient.get<Flow>(`/flows/${flowId}`);
      return response;
    },
    enabled: !!flowId,
  });
};

// Alert hooks
export const useAlerts = (filters: any) => {
  return useQuery({
    queryKey: ['alerts', filters],
    queryFn: async () => {
      const response = await apiClient.get<PaginatedResponse<Alert>>(
        `/alerts?page=${filters.page || 0}&page_size=${filters.pageSize || 50}`
      );
      return response;
    },
  });
};

export const useAlertDetails = (alertId: string) => {
  return useQuery({
    queryKey: ['alerts', alertId],
    queryFn: async () => {
      const response = await apiClient.get<Alert>(`/alerts/${alertId}`);
      return response;
    },
    enabled: !!alertId,
  });
};

// Incident hooks
export const useIncidents = (filters: any) => {
  return useQuery({
    queryKey: ['incidents', filters],
    queryFn: async () => {
      const response = await apiClient.get<PaginatedResponse<Incident>>(
        `/incidents?page=${filters.page || 0}&page_size=${filters.pageSize || 50}`
      );
      return response;
    },
  });
};

export const useIncidentDetails = (incidentId: string) => {
  return useQuery({
    queryKey: ['incidents', incidentId],
    queryFn: async () => {
      const response = await apiClient.get<Incident>(`/incidents/${incidentId}`);
      return response;
    },
    enabled: !!incidentId,
  });
};

// Dashboard metrics
export const useDashboardMetrics = () => {
  return useQuery({
    queryKey: ['dashboard', 'metrics'],
    queryFn: async () => {
      const response = await apiClient.get<any>('/dashboard/metrics');
      return response;
    },
    staleTime: 1 * 60 * 1000, // 1 minute for metrics
  });
};

// Threat timeline
export const useThreatTimeline = () => {
  return useQuery({
    queryKey: ['analytics', 'threat-timeline'],
    queryFn: async () => {
      const response = await apiClient.get<any>('/analytics/threat-timeline');
      return response;
    },
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
};

// Network graph data
export const useNetworkGraph = () => {
  return useQuery({
    queryKey: ['network', 'graph'],
    queryFn: async () => {
      const response = await apiClient.get<any>('/network/graph');
      return response;
    },
    staleTime: 2 * 60 * 1000, // 2 minutes
  });
};

// Network stats
export const useNetworkStats = () => {
  return useQuery({
    queryKey: ['network', 'stats'],
    queryFn: async () => {
      const response = await apiClient.get<any>('/network/stats');
      return response;
    },
    staleTime: 2 * 60 * 1000, // 2 minutes
  });
};

// Attack heatmap data
export const useAttackHeatmap = () => {
  return useQuery({
    queryKey: ['network', 'heatmap'],
    queryFn: async () => {
      const response = await apiClient.get<any>('/network/attack-heatmap');
      return response;
    },
    staleTime: 1 * 60 * 1000, // 1 minute (more real-time)
  });
};
