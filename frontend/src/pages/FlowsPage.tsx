import React, { useState } from 'react';
import {
  Box,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
  Button,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  IconButton,
  Tooltip,
  Grid,
  Typography,
} from '@mui/material';
import { FilterList as FilterListIcon, Close as CloseIcon } from '@mui/icons-material';
import { useFlows } from '@hooks/useApi';
import { useRealtimeFlows } from '@hooks/useRealtime';
import { useFilterStore } from '@stores/filterStore';
import { ErrorBoundary, Loading, ErrorMessage } from '@components/ErrorBoundary';
import { FilterDialog } from '@components/FilterDialog';
import { Toast, useToast } from '@components/Toast';
import { getRiskColor, formatTimestamp } from '@utils/formatters';

const FlowsPage: React.FC = () => {
  const { flowFilters, setFlowFilters } = useFilterStore();
  const { data: flowsData, isLoading, isError } = useFlows(flowFilters);
  const [filterOpen, setFilterOpen] = useState(false);
  const [detailOpen, setDetailOpen] = useState(false);
  const [selectedFlow, setSelectedFlow] = useState<any | null>(null);
  const { toasts, showToast } = useToast();

  // Enable real-time flow synchronization
  useRealtimeFlows();

  if (isLoading) return <Loading />;
  if (isError) return <ErrorMessage message="Failed to load flows" />;

  const flows = (flowsData as any)?.flows || [];
  const totalCount = (flowsData as any)?.total || 0;

  const handleChangePage = (event: unknown, newPage: number) => {
    setFlowFilters({ ...flowFilters, page: newPage });
  };

  const handleChangeRowsPerPage = (event: React.ChangeEvent<HTMLInputElement>) => {
    setFlowFilters({ ...flowFilters, pageSize: parseInt(event.target.value, 10), page: 0 });
  };

  const handleViewDetails = (flow: any) => {
    setSelectedFlow(flow);
    setDetailOpen(true);
  };

  const handleCopyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    showToast('Copied to clipboard', 'success');
  };

  return (
    <ErrorBoundary>
      <Box sx={{ p: 3 }}>
        <Typography variant="h4" sx={{ mb: 3, fontWeight: 600 }}>
          Network Flows
        </Typography>

        {/* Filter Bar */}
        <Box sx={{ display: 'flex', gap: 1, mb: 2, alignItems: 'center' }}>
          <Tooltip title="Open filters">
            <Button
              variant="outlined"
              startIcon={<FilterListIcon />}
              onClick={() => setFilterOpen(true)}
            >
              Filters
            </Button>
          </Tooltip>
          <Typography variant="body2" color="textSecondary">
            Showing {flows.length} of {totalCount} flows
          </Typography>
        </Box>

        {/* Flows Table */}
        <Paper>
          <TableContainer>
            <Table size="small">
              <TableHead>
                <TableRow sx={{ backgroundColor: '#f5f5f5' }}>
                  <TableCell sx={{ fontWeight: 600 }}>Source IP</TableCell>
                  <TableCell sx={{ fontWeight: 600 }}>Dest IP</TableCell>
                  <TableCell sx={{ fontWeight: 600 }} align="right">
                    Sport
                  </TableCell>
                  <TableCell sx={{ fontWeight: 600 }} align="right">
                    Dport
                  </TableCell>
                  <TableCell sx={{ fontWeight: 600 }}>Protocol</TableCell>
                  <TableCell sx={{ fontWeight: 600 }} align="right">
                    Risk
                  </TableCell>
                  <TableCell sx={{ fontWeight: 600 }} align="center">
                    Duration
                  </TableCell>
                  <TableCell sx={{ fontWeight: 600 }} align="center">
                    Actions
                  </TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {flows.map((flow: any, idx: number) => (
                  <TableRow key={idx} hover>
                    <TableCell sx={{ fontFamily: 'monospace', fontSize: '0.85rem' }}>
                      {flow.src_ip}
                    </TableCell>
                    <TableCell sx={{ fontFamily: 'monospace', fontSize: '0.85rem' }}>
                      {flow.dst_ip}
                    </TableCell>
                    <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                      {flow.src_port || '-'}
                    </TableCell>
                    <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                      {flow.dst_port || '-'}
                    </TableCell>
                    <TableCell>
                      <Chip label={flow.protocol?.toUpperCase()} size="small" variant="outlined" />
                    </TableCell>
                    <TableCell align="right">
                      <Chip
                        label={flow.risk_score?.toFixed(1) || '0.0'}
                        size="small"
                        sx={{
                          backgroundColor: getRiskColor(flow.risk_score || 0),
                          color: 'white',
                        }}
                      />
                    </TableCell>
                    <TableCell align="center" sx={{ fontSize: '0.85rem' }}>
                      {flow.duration_ms ? (flow.duration_ms / 1000).toFixed(1) + 's' : '-'}
                    </TableCell>
                    <TableCell align="center">
                      <Tooltip title="View details">
                        <Button
                          size="small"
                          onClick={() => handleViewDetails(flow)}
                        >
                          Details
                        </Button>
                      </Tooltip>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
          <TablePagination
            rowsPerPageOptions={[10, 25, 50, 100]}
            component="div"
            count={totalCount}
            rowsPerPage={flowFilters.pageSize}
            page={flowFilters.page}
            onPageChange={handleChangePage}
            onRowsPerPageChange={handleChangeRowsPerPage}
          />
        </Paper>

        {/* Flow Details Dialog */}
        {selectedFlow && (
          <Dialog open={detailOpen} onClose={() => setDetailOpen(false)} maxWidth="sm" fullWidth>
            <DialogTitle>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                Flow Details
                <IconButton onClick={() => setDetailOpen(false)} size="small">
                  <CloseIcon />
                </IconButton>
              </Box>
            </DialogTitle>
            <DialogContent dividers>
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Typography variant="caption" color="textSecondary">
                    Source IP
                  </Typography>
                  <Typography
                    variant="body2"
                    sx={{
                      fontFamily: 'monospace',
                      cursor: 'pointer',
                      '&:hover': { textDecoration: 'underline' },
                    }}
                    onClick={() => handleCopyToClipboard(selectedFlow.src_ip)}
                  >
                    {selectedFlow.src_ip}
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="caption" color="textSecondary">
                    Dest IP
                  </Typography>
                  <Typography
                    variant="body2"
                    sx={{
                      fontFamily: 'monospace',
                      cursor: 'pointer',
                      '&:hover': { textDecoration: 'underline' },
                    }}
                    onClick={() => handleCopyToClipboard(selectedFlow.dst_ip)}
                  >
                    {selectedFlow.dst_ip}
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="caption" color="textSecondary">
                    Source Port
                  </Typography>
                  <Typography variant="body2">{selectedFlow.src_port || '-'}</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="caption" color="textSecondary">
                    Dest Port
                  </Typography>
                  <Typography variant="body2">{selectedFlow.dst_port || '-'}</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="caption" color="textSecondary">
                    Protocol
                  </Typography>
                  <Typography variant="body2">{selectedFlow.protocol?.toUpperCase()}</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="caption" color="textSecondary">
                    Risk Score
                  </Typography>
                  <Chip label={selectedFlow.risk_score?.toFixed(2)} size="small" />
                </Grid>
                <Grid item xs={12}>
                  <Typography variant="caption" color="textSecondary">
                    Start Time
                  </Typography>
                  <Typography variant="body2">
                    {formatTimestamp(selectedFlow.start_time)}
                  </Typography>
                </Grid>
                <Grid item xs={12}>
                  <Typography variant="caption" color="textSecondary">
                    End Time
                  </Typography>
                  <Typography variant="body2">
                    {formatTimestamp(selectedFlow.end_time)}
                  </Typography>
                </Grid>
              </Grid>
            </DialogContent>
          </Dialog>
        )}

        {/* Filter Dialog */}
        <FilterDialog open={filterOpen} onClose={() => setFilterOpen(false)} type="flows" />

        {/* Toast Notifications */}
        {toasts.map((toast) => (
          <Toast
            key={toast.id}
            open={true}
            onClose={() => {}}
            message={toast.message}
            severity={toast.severity}
          />
        ))}
      </Box>
    </ErrorBoundary>
  );
};

export default FlowsPage;
