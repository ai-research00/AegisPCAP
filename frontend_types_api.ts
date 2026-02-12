"""
Frontend Type Definitions Generated from Backend API Schemas

These types are auto-generated from the Python Pydantic schemas
in src/dashboard/schemas.py to ensure perfect sync between backend and frontend.

Generation tool: python scripts/generate_types.py
Last updated: 2026-02-04
"""

// Type definitions for AegisPCAP Dashboard API
// This file is generated from backend schemas - DO NOT EDIT MANUALLY

// ============================================================================
// Core Domain Types
// ============================================================================

export interface GeoLocation {
  country?: string | null;
  city?: string | null;
  latitude?: number | null;
  longitude?: number | null;
  asn?: string | null;
  organization?: string | null;
}

export interface FlowProtocolInfo {
  protocol: string;
  port?: number | null;
  service?: string | null;
  version?: string | null;
  details: Record<string, any>;
}

// ============================================================================
// Flow Types
// ============================================================================

export interface FlowSummary {
  flow_id: string;
  src_ip: string;
  dst_ip: string;
  src_port: number;
  dst_port: number;
  protocol: string;
  start_time: string; // ISO 8601 datetime
  end_time: string; // ISO 8601 datetime
  duration_seconds: number;
  packets: number;
  bytes: number;
  src_geo?: GeoLocation | null;
  dst_geo?: GeoLocation | null;
  risk_score: number; // 0-100
  status: string;
}

export interface FlowDetail extends FlowSummary {
  packet_size_mean?: number | null;
  packet_size_std?: number | null;
  payload_entropy?: number | null;
  retransmissions?: number | null;
  dns_queries?: string[] | null;
  tls_fingerprint?: string | null;
  user_agent?: string | null;
  features: Record<string, any>;
  alerts: string[]; // Alert IDs
  tags: string[];
}

// ============================================================================
// Alert Types
// ============================================================================

export enum AlertSeverity {
  LOW = "low",
  MEDIUM = "medium",
  HIGH = "high",
  CRITICAL = "critical",
}

export interface AlertBase {
  alert_id: string;
  flow_id: string;
  timestamp: string; // ISO 8601 datetime
  severity: AlertSeverity;
  title: string;
  description: string;
}

export interface AlertSummary extends AlertBase {
  detector: string;
  confidence: number; // 0-1
  affected_ips: string[];
}

export interface AlertDetail extends AlertSummary {
  mitre_techniques: string[];
  evidence: Record<string, any>;
  recommendations: string[];
  related_alerts: string[];
  acknowledged: boolean;
  acknowledged_by?: string | null;
  acknowledged_at?: string | null;
}

// ============================================================================
// Incident Types
// ============================================================================

export enum IncidentStatus {
  OPEN = "open",
  IN_PROGRESS = "in_progress",
  RESOLVED = "resolved",
  CLOSED = "closed",
}

export interface IncidentBase {
  incident_id: string;
  timestamp: string; // ISO 8601 datetime
  status: IncidentStatus;
  title: string;
  description: string;
}

export interface IncidentSummary extends IncidentBase {
  affected_flows: number;
  severity: AlertSeverity;
  alert_count: number;
  risk_score: number; // 0-100
}

export interface IncidentDetail extends IncidentSummary {
  flows: string[]; // Flow IDs
  alerts: string[]; // Alert IDs
  affected_ips: string[];
  attack_vector?: string | null;
  investigation_notes: string;
  assigned_to?: string | null;
  started_at: string; // ISO 8601 datetime
  resolved_at?: string | null;
}

// ============================================================================
// Analytics Types
// ============================================================================

export interface TimeSeriesPoint {
  timestamp: string; // ISO 8601 datetime
  value: number;
  label?: string | null;
}

export interface TimeSeriesData {
  title: string;
  unit: string;
  data_points: TimeSeriesPoint[];
  min_value: number;
  max_value: number;
  avg_value: number;
}

export interface TopItem {
  rank: number;
  label: string;
  value: number;
  percentage: number; // 0-100
}

export interface TopItemsList {
  title: string;
  description?: string | null;
  items: TopItem[];
  total_count: number;
}

export interface SystemStatistics {
  timestamp: string; // ISO 8601 datetime
  total_flows: number;
  total_alerts: number;
  total_incidents: number;
  high_risk_flows: number;
  active_incidents: number;
  average_risk_score: number;
  unique_source_ips: number;
  unique_dest_ips: number;
}

export interface DashboardMetrics {
  stats: SystemStatistics;
  threat_timeline: TimeSeriesData;
  top_source_ips: TopItemsList;
  top_dest_ips: TopItemsList;
  top_protocols: TopItemsList;
  alerts_by_severity: Record<string, number>;
  incidents_by_status: Record<string, number>;
}

// ============================================================================
// Dashboard Overview
// ============================================================================

export interface DashboardOverview {
  timestamp: string; // ISO 8601 datetime
  status: string;
  metrics: DashboardMetrics;
  recent_alerts: AlertSummary[];
  recent_incidents: IncidentSummary[];
  high_risk_flows: FlowSummary[];
}

// ============================================================================
// Network Topology
// ============================================================================

export interface NetworkNode {
  id: string;
  label: string;
  type: "internal" | "external" | "server" | "client";
  ip_address: string;
  risk_score: number;
  flow_count: number;
  alert_count: number;
  geo_location?: GeoLocation | null;
  color?: string | null;
}

export interface NetworkLink {
  source_id: string;
  target_id: string;
  flow_count: number;
  bytes_transferred: number;
  alert_count: number;
  risk_score: number;
  color?: string | null;
}

export interface NetworkTopology {
  nodes: NetworkNode[];
  links: NetworkLink[];
  timestamp: string; // ISO 8601 datetime
  summary: Record<string, any>;
}

// ============================================================================
// Threat Intelligence
// ============================================================================

export interface ThreatIndicator {
  type: "ip" | "domain" | "hash" | "url";
  value: string;
  severity: AlertSeverity;
  source: string;
  first_seen: string; // ISO 8601 datetime
  last_seen: string; // ISO 8601 datetime
  matched_flows: number;
}

export interface ThreatIntelligenceSummary {
  timestamp: string; // ISO 8601 datetime
  total_indicators: number;
  indicators_by_type: Record<string, number>;
  recent_indicators: ThreatIndicator[];
}

// ============================================================================
// Pagination & Response Wrappers
// ============================================================================

export interface PaginationInfo {
  page: number;
  page_size: number;
  total: number;
  pages: number;
}

export interface PaginatedResponse<T> {
  data: T[];
  pagination: PaginationInfo;
  meta?: Record<string, any>;
}

// ============================================================================
// Error Types
// ============================================================================

export interface ErrorResponse {
  error: string;
  message: string;
  timestamp: string; // ISO 8601 datetime
  request_id?: string | null;
}

// ============================================================================
// Filter Types
// ============================================================================

export interface FlowFilters {
  src_ip?: string;
  dst_ip?: string;
  src_port?: number;
  dst_port?: number;
  protocol?: string;
  min_risk_score?: number;
  max_risk_score?: number;
  start_time?: string;
  end_time?: string;
  has_alerts?: boolean;
}

export interface AlertFilters {
  severity?: AlertSeverity;
  detector?: string;
  flow_id?: string;
  min_confidence?: number;
  acknowledged?: boolean;
  start_time?: string;
  end_time?: string;
}

export interface IncidentFilters {
  status?: IncidentStatus;
  severity?: AlertSeverity;
  assigned_to?: string;
  min_risk_score?: number;
  start_time?: string;
  end_time?: string;
}

// ============================================================================
// WebSocket Message Types
// ============================================================================

export enum WebSocketMessageType {
  ALERT = "alert",
  INCIDENT = "incident",
  FLOW_UPDATE = "flow_update",
  STATISTICS = "statistics",
  TOPOLOGY_UPDATE = "topology_update",
  HEARTBEAT = "heartbeat",
}

export interface WebSocketMessage<T = any> {
  type: WebSocketMessageType;
  timestamp: string; // ISO 8601 datetime
  data: T;
}

export interface AlertMessage extends WebSocketMessage {
  type: WebSocketMessageType.ALERT;
  data: {
    alert_id: string;
    severity: AlertSeverity;
    title: string;
    timestamp: string;
    flow_id: string;
  };
}

export interface IncidentMessage extends WebSocketMessage {
  type: WebSocketMessageType.INCIDENT;
  data: {
    incident_id: string;
    title: string;
    severity: AlertSeverity;
    timestamp: string;
    affected_flows: number;
  };
}

export interface FlowUpdateMessage extends WebSocketMessage {
  type: WebSocketMessageType.FLOW_UPDATE;
  data: {
    flow_id: string;
    src_ip: string;
    dst_ip: string;
    risk_score: number;
    packet_count: number;
    byte_count: number;
  };
}

export interface StatisticsMessage extends WebSocketMessage {
  type: WebSocketMessageType.STATISTICS;
  data: {
    total_flows: number;
    total_alerts: number;
    total_incidents: number;
    active_threats: number;
    timestamp: string;
  };
}

// ============================================================================
// Helper Type Guards
// ============================================================================

export function isAlertMessage(msg: any): msg is AlertMessage {
  return msg.type === WebSocketMessageType.ALERT;
}

export function isIncidentMessage(msg: any): msg is IncidentMessage {
  return msg.type === WebSocketMessageType.INCIDENT;
}

export function isFlowUpdateMessage(msg: any): msg is FlowUpdateMessage {
  return msg.type === WebSocketMessageType.FLOW_UPDATE;
}

export function isStatisticsMessage(msg: any): msg is StatisticsMessage {
  return msg.type === WebSocketMessageType.STATISTICS;
}

// ============================================================================
// Union Types (for discriminated unions in reducers/handlers)
// ============================================================================

export type AnyWebSocketMessage =
  | AlertMessage
  | IncidentMessage
  | FlowUpdateMessage
  | StatisticsMessage
  | WebSocketMessage;

export type AnyFlow = FlowSummary | FlowDetail;
export type AnyAlert = AlertSummary | AlertDetail;
export type AnyIncident = IncidentSummary | IncidentDetail;
