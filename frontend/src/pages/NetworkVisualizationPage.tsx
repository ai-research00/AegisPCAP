import React, { useState } from 'react';
import {
  Box,
  Grid,
  Paper,
  Typography,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  IconButton,
  Card,
  CardContent,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
} from '@mui/material';
import { Close as CloseIcon } from '@mui/icons-material';
import { useNetworkGraph, useNetworkStats, useAttackHeatmap } from '@hooks/useApi';
import { ErrorBoundary, Loading, ErrorMessage } from '@components/ErrorBoundary';
import NetworkGraphComponent from '@components/NetworkGraph';
import AttackHeatmap from '@components/AttackHeatmap';
import type { NetworkNode, NetworkStats } from '../types/network';
import { getRiskColor, formatNumber } from '@utils/formatters';

const NetworkVisualizationPage: React.FC = () => {
  const { data: graphData, isLoading: graphLoading, isError: graphError } = useNetworkGraph();
  const { data: statsData, isLoading: statsLoading } = useNetworkStats();
  const { data: heatmapData, isLoading: heatmapLoading } = useAttackHeatmap();

  const [detailOpen, setDetailOpen] = useState(false);
  const [selectedNode, setSelectedNode] = useState<NetworkNode | null>(null);

  const handleNodeClick = (node: NetworkNode) => {
    setSelectedNode(node);
    setDetailOpen(true);
  };

  if (graphLoading) return <Loading />;
  if (graphError) return <ErrorMessage message="Failed to load network data" />;

  const networkGraph = graphData || { nodes: [], edges: [] };
  const stats = statsData as NetworkStats | undefined;
  const heatmap = heatmapData || [];

  return (
    <ErrorBoundary>
      <Box sx={{ p: 3 }}>
        <Typography variant="h4" sx={{ mb: 3, fontWeight: 600 }}>
          Network Topology & Attack Visualization
        </Typography>

        {/* Network Graph */}
        <Paper sx={{ mb: 3, p: 2 }}>
          <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
            Network Graph
          </Typography>
          <Box sx={{ position: 'relative', width: '100%', minHeight: 600 }}>
            <NetworkGraphComponent
              data={networkGraph}
              onNodeClick={handleNodeClick}
              height={600}
              interactive={true}
            />
          </Box>
          <Typography variant="caption" color="textSecondary" sx={{ mt: 1, display: 'block' }}>
            Drag nodes to reposition. Node size indicates flow count. Color indicates risk level.
          </Typography>
        </Paper>

        {/* Attack Heatmap */}
        {!heatmapLoading && (
          <Paper sx={{ mb: 3 }}>
            <AttackHeatmap data={heatmap} />
          </Paper>
        )}

        {/* Statistics Grid */}
        {stats && (
          <Grid container spacing={3} sx={{ mb: 3 }}>
            <Grid item xs={12} sm={6} md={3}>
              <Card>
                <CardContent>
                  <Typography color="textSecondary" gutterBottom>
                    Total Nodes
                  </Typography>
                  <Typography variant="h5" sx={{ fontWeight: 600 }}>
                    {formatNumber(stats.totalNodes || 0)}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Card>
                <CardContent>
                  <Typography color="textSecondary" gutterBottom>
                    Total Connections
                  </Typography>
                  <Typography variant="h5" sx={{ fontWeight: 600 }}>
                    {formatNumber(stats.totalEdges || 0)}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Card>
                <CardContent>
                  <Typography color="textSecondary" gutterBottom>
                    Avg Risk Score
                  </Typography>
                  <Typography variant="h5" sx={{ fontWeight: 600 }}>
                    {(stats.avgRiskScore || 0).toFixed(1)}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Card sx={{ background: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)' }}>
                <CardContent>
                  <Typography color="textSecondary" sx={{ color: 'rgba(255,255,255,0.7)' }} gutterBottom>
                    High Risk Nodes
                  </Typography>
                  <Typography variant="h5" sx={{ color: 'white', fontWeight: 600 }}>
                    {stats.topRiskyNodes?.length || 0}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        )}

        {/* Top Risky Nodes Table */}
        {stats && stats.topRiskyNodes && stats.topRiskyNodes.length > 0 && (
          <Paper sx={{ mb: 3 }}>
            <TableContainer>
              <Table size="small">
                <TableHead>
                  <TableRow sx={{ backgroundColor: '#f5f5f5' }}>
                    <TableCell sx={{ fontWeight: 600 }}>IP Address</TableCell>
                    <TableCell sx={{ fontWeight: 600 }} align="right">
                      Risk Score
                    </TableCell>
                    <TableCell sx={{ fontWeight: 600 }} align="right">
                      Flow Count
                    </TableCell>
                    <TableCell sx={{ fontWeight: 600 }}>Type</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {stats.topRiskyNodes.map((node, idx) => (
                    <TableRow
                      key={idx}
                      hover
                      sx={{ cursor: 'pointer' }}
                      onClick={() => handleNodeClick(node)}
                    >
                      <TableCell sx={{ fontFamily: 'monospace' }}>{node.id}</TableCell>
                      <TableCell align="right">
                        <Chip
                          label={node.riskScore.toFixed(1)}
                          size="small"
                          sx={{ backgroundColor: getRiskColor(node.riskScore), color: 'white' }}
                        />
                      </TableCell>
                      <TableCell align="right">{formatNumber(node.flowCount)}</TableCell>
                      <TableCell>
                        <Chip label={node.type} size="small" variant="outlined" />
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Paper>
        )}

        {/* Top Attack Paths */}
        {stats && stats.topAttackPaths && stats.topAttackPaths.length > 0 && (
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
              Top Attack Paths
            </Typography>
            <TableContainer>
              <Table size="small">
                <TableHead>
                  <TableRow sx={{ backgroundColor: '#f5f5f5' }}>
                    <TableCell sx={{ fontWeight: 600 }}>Source</TableCell>
                    <TableCell sx={{ fontWeight: 600 }} align="center">
                      →
                    </TableCell>
                    <TableCell sx={{ fontWeight: 600 }}>Destination</TableCell>
                    <TableCell sx={{ fontWeight: 600 }} align="right">
                      Attack Count
                    </TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {stats.topAttackPaths.map((path, idx) => (
                    <TableRow key={idx} hover>
                      <TableCell sx={{ fontFamily: 'monospace', fontSize: '0.85rem' }}>
                        {path.source}
                      </TableCell>
                      <TableCell align="center">→</TableCell>
                      <TableCell sx={{ fontFamily: 'monospace', fontSize: '0.85rem' }}>
                        {path.target}
                      </TableCell>
                      <TableCell align="right">{formatNumber(path.attacks)}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Paper>
        )}
      </Box>

      {/* Node Details Dialog */}
      {selectedNode && (
        <Dialog open={detailOpen} onClose={() => setDetailOpen(false)} maxWidth="sm" fullWidth>
          <DialogTitle>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              Node Details
              <IconButton onClick={() => setDetailOpen(false)} size="small">
                <CloseIcon />
              </IconButton>
            </Box>
          </DialogTitle>
          <DialogContent dividers>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <Box>
                <Typography variant="caption" color="textSecondary">
                  IP Address
                </Typography>
                <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                  {selectedNode.id}
                </Typography>
              </Box>

              <Box>
                <Typography variant="caption" color="textSecondary">
                  Risk Score
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 0.5 }}>
                  <Chip
                    label={selectedNode.riskScore.toFixed(2)}
                    sx={{ backgroundColor: getRiskColor(selectedNode.riskScore), color: 'white' }}
                  />
                  <Typography variant="body2" color="textSecondary">
                    {selectedNode.riskScore > 80
                      ? 'Critical'
                      : selectedNode.riskScore > 60
                      ? 'High'
                      : selectedNode.riskScore > 40
                      ? 'Medium'
                      : 'Low'}
                  </Typography>
                </Box>
              </Box>

              <Box>
                <Typography variant="caption" color="textSecondary">
                  Flow Count
                </Typography>
                <Typography variant="body2">{formatNumber(selectedNode.flowCount)}</Typography>
              </Box>

              <Box>
                <Typography variant="caption" color="textSecondary">
                  Type
                </Typography>
                <Chip label={selectedNode.type} size="small" variant="outlined" />
              </Box>

              {selectedNode.geoLocation && (
                <Box>
                  <Typography variant="caption" color="textSecondary">
                    Location
                  </Typography>
                  <Typography variant="body2">{selectedNode.geoLocation.country}</Typography>
                  <Typography variant="caption" color="textSecondary">
                    ({selectedNode.geoLocation.latitude.toFixed(2)}, {selectedNode.geoLocation.longitude.toFixed(2)})
                  </Typography>
                </Box>
              )}
            </Box>
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setDetailOpen(false)}>Close</Button>
          </DialogActions>
        </Dialog>
      )}
    </ErrorBoundary>
  );
};

export default NetworkVisualizationPage;
