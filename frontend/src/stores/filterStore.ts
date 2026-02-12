import { create } from 'zustand';

export interface FlowFiltersState {
  sourceIp?: string;
  destIp?: string;
  protocol?: string;
  minRiskScore?: number;
  maxRiskScore?: number;
  page: number;
  pageSize: number;
  sortBy: string;
  sortOrder: 'asc' | 'desc';
}

export interface AlertFiltersState {
  severity?: 'low' | 'medium' | 'high' | 'critical';
  detector?: string;
  acknowledged?: boolean;
  page: number;
  pageSize: number;
  sortBy: string;
  sortOrder: 'asc' | 'desc';
}

export interface IncidentFiltersState {
  status?: 'open' | 'investigating' | 'resolved' | 'closed';
  assignedTo?: string;
  page: number;
  pageSize: number;
  sortBy: string;
  sortOrder: 'asc' | 'desc';
}

interface FilterStore {
  flowFilters: FlowFiltersState;
  alertFilters: AlertFiltersState;
  incidentFilters: IncidentFiltersState;
  
  setFlowFilters: (filters: Partial<FlowFiltersState>) => void;
  resetFlowFilters: () => void;
  
  setAlertFilters: (filters: Partial<AlertFiltersState>) => void;
  resetAlertFilters: () => void;
  
  setIncidentFilters: (filters: Partial<IncidentFiltersState>) => void;
  resetIncidentFilters: () => void;
}

const defaultFlowFilters: FlowFiltersState = {
  page: 0,
  pageSize: 50,
  sortBy: 'start_time',
  sortOrder: 'desc',
};

const defaultAlertFilters: AlertFiltersState = {
  page: 0,
  pageSize: 50,
  sortBy: 'timestamp',
  sortOrder: 'desc',
};

const defaultIncidentFilters: IncidentFiltersState = {
  page: 0,
  pageSize: 50,
  sortBy: 'created_at',
  sortOrder: 'desc',
};

export const useFilterStore = create<FilterStore>((set) => ({
  flowFilters: defaultFlowFilters,
  alertFilters: defaultAlertFilters,
  incidentFilters: defaultIncidentFilters,

  setFlowFilters: (filters) =>
    set((state) => ({
      flowFilters: { ...state.flowFilters, ...filters },
    })),
  resetFlowFilters: () =>
    set(() => ({
      flowFilters: defaultFlowFilters,
    })),

  setAlertFilters: (filters) =>
    set((state) => ({
      alertFilters: { ...state.alertFilters, ...filters },
    })),
  resetAlertFilters: () =>
    set(() => ({
      alertFilters: defaultAlertFilters,
    })),

  setIncidentFilters: (filters) =>
    set((state) => ({
      incidentFilters: { ...state.incidentFilters, ...filters },
    })),
  resetIncidentFilters: () =>
    set(() => ({
      incidentFilters: defaultIncidentFilters,
    })),
}));
