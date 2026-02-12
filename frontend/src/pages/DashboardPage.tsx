import React, { useState } from 'react';
import {
  Box,
  Grid,
  Paper,
  Typography,
  Card,
  CardContent,
  LinearProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
} from '@mui/material';
import { LineChart, Line, PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { useDashboardMetrics, useThreatTimeline } from '@hooks/useApi';
import { useRealtimeStatistics, useAlertNotification } from '@hooks/useRealtime';
import { ErrorBoundary, Loading, ErrorMessage } from '@components/ErrorBoundary';
import { Toast } from '@components/Toast';
import { getRiskColor, getSeverityColor, formatNumber } from '@utils/formatters';

const DashboardPage: React.FC = () => {
  const { data: metrics, isLoading: metricsLoading, isError: metricsError } = useDashboardMetrics();
  const { data: timeline, isLoading: timelineLoading } = useThreatTimeline();
  const [toastOpen, setToastOpen] = useState(false);

  // Enable real-time updates
  useRealtimeStatistics();
  useAlertNotification();

  if (metricsLoading) return <Loading />;
  if (metricsError) return <ErrorMessage />;

  const kpis = (metrics as any)?.kpis || {};
  const topAttackers = (metrics as any)?.topAttackers || [];
  const protocolDistribution = (metrics as any)?.protocolDistribution || [];

  return (
    <ErrorBoundary>
      <Box sx={{ p: 3 }}>
        <Typography variant="h4" sx={{ mb: 3, fontWeight: 600 }}>
          Dashboard
        </Typography>

        {/* KPI Cards */}
        <Grid container spacing={3} sx={{ mb: 4 }}>
          <Grid item xs={12} sm={6} md={3}>
            <Card sx={{ background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' }}>
              <CardContent>
                <Typography color="textSecondary" sx={{ color: 'rgba(255,255,255,0.7)' }}>
                  Total Flows
                </Typography>
                <Typography variant="h5" sx={{ color: 'white', mt: 1, fontWeight: 600 }}>
                  {formatNumber(kpis.totalFlows || 0)}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card sx={{ background: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)' }}>
              <CardContent>
                <Typography color="textSecondary" sx={{ color: 'rgba(255,255,255,0.7)' }}>
                  Active Threats
                </Typography>
                <Typography variant="h5" sx={{ color: 'white', mt: 1, fontWeight: 600 }}>
                  {kpis.activeThreats || 0}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card sx={{ background: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)' }}>
              <CardContent>
                <Typography color="textSecondary" sx={{ color: 'rgba(255,255,255,0.7)' }}>
                  Incidents
                </Typography>
                <Typography variant="h5" sx={{ color: 'white', mt: 1, fontWeight: 600 }}>
                  {kpis.incidents || 0}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card sx={{ background: 'linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)' }}>
              <CardContent>
                <Typography color="textSecondary" sx={{ color: 'rgba(255,255,255,0.7)' }}>
                  Avg Risk Score
                </Typography>
                <Typography variant="h5" sx={{ color: 'white', mt: 1, fontWeight: 600 }}>
                  {(kpis.avgRiskScore || 0).toFixed(1)}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>

        {/* Charts */}
        <Grid container spacing={3} sx={{ mb: 4 }}>
          {/* Threat Timeline */}
          <Grid item xs={12} md={8}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
                Threat Timeline (24h)
              </Typography>
              {timelineLoading ? (
                <Loading size="small" />
              ) : (
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={timeline || []}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" />
                    <YAxis />
                    <Tooltip />
                    <Line type="monotone" dataKey="threats" stroke="#f5576c" strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              )}
            </Paper>
          </Grid>

          {/* Protocol Distribution */}
          <Grid item xs={12} md={4}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
                Protocol Distribution
              </Typography>
              {protocolDistribution.length > 0 ? (
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={protocolDistribution}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {protocolDistribution.map((_: any, index: number) => (
                        <Cell key={`cell-${index}`} fill={['#667eea', '#764ba2', '#f093fb', '#f5576c'][index % 4]} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              ) : (
                <Typography color="textSecondary">No data available</Typography>
              )}
            </Paper>
          </Grid>
        </Grid>

        {/* Top Attackers */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
              Top Attacking IPs
            </Typography>
            {topAttackers.length > 0 ? (
              <TableContainer>
                <Table size="small">
                  <TableHead>
                    <TableRow sx={{ backgroundColor: '#f5f5f5' }}>
                      <TableCell sx={{ fontWeight: 600 }}>IP Address</TableCell>
                      <TableCell sx={{ fontWeight: 600 }} align="right">
                        Flows
                      </TableCell>
                      <TableCell sx={{ fontWeight: 600 }} align="right">
                        Risk Score
                      </TableCell>
                      <TableCell sx={{ fontWeight: 600 }} align="right">
                        Threat Level
                      </TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {topAttackers.map((attacker: any, idx: number) => (
                      <TableRow key={idx} hover>
                        <TableCell sx={{ fontFamily: 'monospace' }}>{attacker.ip}</TableCell>
                        <TableCell align="right">{attacker.flowCount}</TableCell>
                        <TableCell align="right">
                          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: 1 }}>
                            <LinearProgress
                              variant="determinate"
                              value={attacker.riskScore}
                              sx={{ width: 60, mr: 1 }}
                            />
                            {attacker.riskScore.toFixed(1)}
                          </Box>
                        </TableCell>
                        <TableCell align="right">
                          <Chip
                            label={attacker.threatLevel}
                            size="small"
                            sx={{ backgroundColor: getSeverityColor(attacker.threatLevel) }}
                          />
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            ) : (
              <Typography color="textSecondary">No data available</Typography>
            )}
          </Paper>
        </Grid>
      </Box>
      <Toast
        open={toastOpen}
        onClose={() => setToastOpen(false)}
        message="Dashboard updated"
        severity="info"
      />
    </ErrorBoundary>
  );
};

export default DashboardPage;
