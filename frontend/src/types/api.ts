// GeoLocation
export interface GeoLocation {
  country: string;
  country_code: string;
  city: string;
  latitude: number;
  longitude: number;
  asn: string;
}

// Flow types
export interface FlowProtocolInfo {
  protocol: string;
  src_port: number;
  dst_port: number;
  flags?: string;
}

export interface Flow {
  id: string;
  src_ip: string;
  dst_ip: string;
  protocol_info: FlowProtocolInfo;
  src_geo: GeoLocation;
  dst_geo: GeoLocation;
  start_time: string;
  end_time: string;
  packets_sent: number;
  packets_recv: number;
  bytes_sent: number;
  bytes_recv: number;
  duration_seconds: number;
  risk_score: number;
  alerts: Alert[];
  features?: Record<string, number>;
  tags: string[];
}

export interface FlowSummary {
  id: string;
  src_ip: string;
  dst_ip: string;
  protocol: string;
  src_port: number;
  dst_port: number;
  start_time: string;
  duration_seconds: number;
  packets: number;
  bytes: number;
  risk_score: number;
}

// Alert types
export enum AlertSeverity {
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high',
  CRITICAL = 'critical',
}

export interface MITREMapping {
  tactic: string;
  technique: string;
  technique_id: string;
}

export interface Alert {
  id: string;
  flow_id: string;
  detector: string;
  severity: AlertSeverity;
  message: string;
  timestamp: string;
  mitre_techniques: MITREMapping[];
  evidence: Record<string, any>;
  acknowledged: boolean;
  acknowledged_by?: string;
  acknowledged_at?: string;
}

export interface AlertDetail extends Alert {
  flow: Flow;
  related_flows: Flow[];
  recommendation: string;
}

// Incident types
export enum IncidentStatus {
  OPEN = 'open',
  INVESTIGATING = 'investigating',
  RESOLVED = 'resolved',
  CLOSED = 'closed',
}

export interface Incident {
  id: string;
  title: string;
  description: string;
  status: IncidentStatus;
  severity: AlertSeverity;
  created_at: string;
  updated_at: string;
  assigned_to?: string;
  flows: Flow[];
  alerts: Alert[];
  mitre_techniques: MITREMapping[];
  evidence: Record<string, any>;
}

// Analytics types
export interface TimeSeriesPoint {
  timestamp: string;
  value: number;
  label?: string;
}

export interface TimeSeriesData {
  title: string;
  data: TimeSeriesPoint[];
}

export interface TopItem {
  name: string;
  value: number;
  percentage: number;
}

export interface TopItemsList {
  title: string;
  items: TopItem[];
}

export interface SystemStatistics {
  total_flows: number;
  total_alerts: number;
  total_incidents: number;
  avg_risk_score: number;
  critical_alerts: number;
  high_alerts: number;
  open_incidents: number;
}

export interface DashboardMetrics {
  timestamp: string;
  statistics: SystemStatistics;
  threat_timeline: TimeSeriesData;
  top_attacking_ips: TopItemsList;
  top_targeted_services: TopItemsList;
  protocol_distribution: TopItemsList;
}

// Topology
export interface NetworkNode {
  id: string;
  label: string;
  type: 'source' | 'destination';
  risk_level: 'low' | 'medium' | 'high' | 'critical';
  geo?: GeoLocation;
}

export interface NetworkLink {
  source: string;
  target: string;
  weight: number;
  risk_level: 'low' | 'medium' | 'high' | 'critical';
}

export interface NetworkTopology {
  nodes: NetworkNode[];
  links: NetworkLink[];
}

// Pagination
export interface PaginationInfo {
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

export interface PaginatedResponse<T> {
  data: T[];
  pagination: PaginationInfo;
}

// Errors
export interface ErrorResponse {
  error_code: string;
  detail: string;
  status_code: number;
}

// Filters
export interface FlowFilters {
  src_ip?: string;
  dst_ip?: string;
  protocol?: string;
  risk_score_min?: number;
  risk_score_max?: number;
  start_time?: string;
  end_time?: string;
  page?: number;
  page_size?: number;
  sort_by?: string;
  sort_order?: 'asc' | 'desc';
}

export interface AlertFilters {
  severity?: AlertSeverity;
  detector?: string;
  acknowledged?: boolean;
  start_time?: string;
  end_time?: string;
  page?: number;
  page_size?: number;
  sort_by?: string;
  sort_order?: 'asc' | 'desc';
}

export interface IncidentFilters {
  status?: IncidentStatus;
  severity?: AlertSeverity;
  assigned_to?: string;
  start_time?: string;
  end_time?: string;
  page?: number;
  page_size?: number;
  sort_by?: string;
  sort_order?: 'asc' | 'desc';
}

// WebSocket message types
export enum WebSocketMessageType {
  ALERT = 'alert',
  INCIDENT = 'incident',
  FLOW = 'flow',
  TOPOLOGY = 'topology',
  STATISTICS = 'statistics',
  HEARTBEAT = 'heartbeat',
}

export interface WebSocketMessage<T = any> {
  type: WebSocketMessageType;
  data: T;
  timestamp: string;
}

// Type guards
export function isAlertMessage(msg: WebSocketMessage): msg is WebSocketMessage<Alert> {
  return msg.type === WebSocketMessageType.ALERT;
}

export function isIncidentMessage(msg: WebSocketMessage): msg is WebSocketMessage<Incident> {
  return msg.type === WebSocketMessageType.INCIDENT;
}

export function isFlowMessage(msg: WebSocketMessage): msg is WebSocketMessage<Flow> {
  return msg.type === WebSocketMessageType.FLOW;
}

export function isStatisticsMessage(msg: WebSocketMessage): msg is WebSocketMessage<SystemStatistics> {
  return msg.type === WebSocketMessageType.STATISTICS;
}
