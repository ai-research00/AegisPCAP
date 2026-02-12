// Network visualization types for D3.js

export interface NetworkNode {
  id: string;
  label: string;
  type: 'source' | 'destination' | 'both';
  riskScore: number;
  flowCount: number;
  geoLocation?: {
    country: string;
    latitude: number;
    longitude: number;
  };
}

export interface NetworkEdge {
  source: string;
  target: string;
  weight: number;
  protocol: string;
  riskScore: number;
  flowCount: number;
  bandwidth?: number;
}

export interface NetworkGraph {
  nodes: NetworkNode[];
  edges: NetworkEdge[];
}

export interface D3Node extends NetworkNode {
  x?: number;
  y?: number;
  vx?: number;
  vy?: number;
  index?: number;
  fx?: number | null;
  fy?: number | null;
}

export interface D3Link {
  source: string | D3Node;
  target: string | D3Node;
  weight: number;
  protocol: string;
  riskScore: number;
  flowCount: number;
  bandwidth?: number;
}

export interface AttackHeatmapData {
  timestamp: string;
  attackCount: number;
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
}

export interface NetworkStats {
  totalNodes: number;
  totalEdges: number;
  avgRiskScore: number;
  topRiskyNodes: NetworkNode[];
  topAttackPaths: Array<{
    source: string;
    target: string;
    attacks: number;
  }>;
}
