import React, { useMemo } from 'react';
import { Box, Paper, Typography, Grid, Chip } from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import type { AttackHeatmapData } from '../types/network';
import { getSeverityColor } from '@utils/formatters';

interface AttackHeatmapProps {
  data: AttackHeatmapData[];
  width?: number;
  height?: number;
}

export const AttackHeatmap: React.FC<AttackHeatmapProps> = ({ data, width = 1200, height = 300 }) => {
  const chartData = useMemo(
    () =>
      data.map((d) => ({
        ...d,
        timestamp: new Date(d.timestamp).toLocaleTimeString(),
      })),
    [data]
  );

  return (
    <Paper sx={{ p: 2 }}>
      <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
        Attack Timeline (24h)
      </Typography>
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="timestamp" />
          <YAxis />
          <Tooltip
            contentStyle={{
              backgroundColor: '#fff',
              border: '1px solid #ccc',
              borderRadius: 4,
            }}
            formatter={(value: any) => [value, 'Attacks']}
          />
          <Bar
            dataKey="attackCount"
            radius={[8, 8, 0, 0]}
            fill="#f44336"
            opacity={0.8}
            name="Attacks"
          />
        </BarChart>
      </ResponsiveContainer>

      {/* Legend */}
      <Box sx={{ mt: 2, display: 'flex', gap: 2, flexWrap: 'wrap' }}>
        <Typography variant="caption" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Box sx={{ width: 16, height: 16, backgroundColor: '#4caf50', borderRadius: 1 }} />
          Low
        </Typography>
        <Typography variant="caption" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Box sx={{ width: 16, height: 16, backgroundColor: '#ffc107', borderRadius: 1 }} />
          Medium
        </Typography>
        <Typography variant="caption" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Box sx={{ width: 16, height: 16, backgroundColor: '#ff9800', borderRadius: 1 }} />
          High
        </Typography>
        <Typography variant="caption" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Box sx={{ width: 16, height: 16, backgroundColor: '#d32f2f', borderRadius: 1 }} />
          Critical
        </Typography>
      </Box>
    </Paper>
  );
};

export default AttackHeatmap;
