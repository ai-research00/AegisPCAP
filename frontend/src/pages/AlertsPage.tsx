import React, { useState } from 'react';
import {
  Box,
  Paper,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  Button,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  IconButton,
  Tooltip,
  Grid,
  Typography,
  ButtonGroup,
} from '@mui/material';
import { FilterList as FilterListIcon, Close as CloseIcon, CheckCircle as CheckCircleIcon } from '@mui/icons-material';
import { useAlerts, useAlertDetails } from '@hooks/useApi';
import { useRealtimeAlerts } from '@hooks/useRealtime';
import { useFilterStore } from '@stores/filterStore';
import { ErrorBoundary, Loading, ErrorMessage } from '@components/ErrorBoundary';
import { FilterDialog } from '@components/FilterDialog';
import { Toast, useToast } from '@components/Toast';
import { getSeverityColor, formatTimestamp } from '@utils/formatters';

const AlertsPage: React.FC = () => {
  const { alertFilters, setAlertFilters } = useFilterStore();
  const { data: alertsData, isLoading, isError } = useAlerts(alertFilters);
  const [filterOpen, setFilterOpen] = useState(false);
  const [detailOpen, setDetailOpen] = useState(false);
  const [selectedAlertId, setSelectedAlertId] = useState<string | null>(null);
  const { data: alertDetail } = useAlertDetails(selectedAlertId || '');
  const { toasts, showToast } = useToast();

  // Enable real-time alert synchronization
  useRealtimeAlerts();

  if (isLoading) return <Loading />;
  if (isError) return <ErrorMessage message="Failed to load alerts" />;

  const alerts = (alertsData as any)?.alerts || [];
  const totalCount = (alertsData as any)?.total || 0;

  const handleViewDetails = (alertId: string) => {
    setSelectedAlertId(alertId);
    setDetailOpen(true);
  };

  const handleAcknowledge = async (alertId: string) => {
    try {
      // API call would go here
      showToast('Alert acknowledged', 'success');
    } catch (error) {
      showToast('Failed to acknowledge alert', 'error');
    }
  };

  const handleSeverityFilter = (severity: string) => {
    setAlertFilters({ ...alertFilters, severity: severity as any, page: 0 });
  };

  return (
    <ErrorBoundary>
      <Box sx={{ p: 3 }}>
        <Typography variant="h4" sx={{ mb: 3, fontWeight: 600 }}>
          Security Alerts
        </Typography>

        {/* Filters */}
        <Box sx={{ display: 'flex', gap: 2, mb: 3, alignItems: 'center', flexWrap: 'wrap' }}>
          <Tooltip title="Open filters">
            <Button
              variant="outlined"
              startIcon={<FilterListIcon />}
              onClick={() => setFilterOpen(true)}
            >
              More Filters
            </Button>
          </Tooltip>

          <ButtonGroup variant="outlined" size="small">
            <Button
              onClick={() => handleSeverityFilter('')}
              variant={alertFilters.severity ? 'outlined' : 'contained'}
            >
              All
            </Button>
            <Button
              onClick={() => handleSeverityFilter('low')}
              variant={alertFilters.severity === 'low' ? 'contained' : 'outlined'}
            >
              Low
            </Button>
            <Button
              onClick={() => handleSeverityFilter('medium')}
              variant={alertFilters.severity === 'medium' ? 'contained' : 'outlined'}
            >
              Medium
            </Button>
            <Button
              onClick={() => handleSeverityFilter('high')}
              variant={alertFilters.severity === 'high' ? 'contained' : 'outlined'}
            >
              High
            </Button>
            <Button
              onClick={() => handleSeverityFilter('critical')}
              variant={alertFilters.severity === 'critical' ? 'contained' : 'outlined'}
            >
              Critical
            </Button>
          </ButtonGroup>

          <Typography variant="body2" color="textSecondary">
            Showing {alerts.length} of {totalCount} alerts
          </Typography>
        </Box>

        {/* Alerts List */}
        <Paper>
          {alerts.length > 0 ? (
            <List sx={{ maxHeight: '70vh', overflow: 'auto' }}>
              {alerts.map((alert: any, idx: number) => (
                <ListItemButton
                  key={idx}
                  onClick={() => handleViewDetails(alert.id)}
                  sx={{ borderBottom: '1px solid #eee' }}
                >
                  <ListItemText
                    primary={
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Chip
                          label={alert.severity?.toUpperCase()}
                          size="small"
                          sx={{ backgroundColor: getSeverityColor(alert.severity || 'low'), color: 'white' }}
                        />
                        <Typography variant="subtitle2" sx={{ flex: 1 }}>
                          {alert.detector}
                        </Typography>
                        {!alert.acknowledged && (
                          <Chip label="New" size="small" color="primary" variant="filled" />
                        )}
                      </Box>
                    }
                    secondary={
                      <Box sx={{ display: 'flex', gap: 1, justifyContent: 'space-between', mt: 0.5 }}>
                        <Typography variant="caption" sx={{ fontFamily: 'monospace', color: '#666' }}>
                          {alert.src_ip} → {alert.dst_ip}
                        </Typography>
                        <Typography variant="caption" color="textSecondary">
                          {formatTimestamp(alert.timestamp)}
                        </Typography>
                      </Box>
                    }
                  />
                  <Tooltip title="Acknowledge">
                    <IconButton
                      size="small"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleAcknowledge(alert.id);
                      }}
                    >
                      <CheckCircleIcon />
                    </IconButton>
                  </Tooltip>
                </ListItemButton>
              ))}
            </List>
          ) : (
            <Box sx={{ p: 3, textAlign: 'center' }}>
              <Typography color="textSecondary">No alerts found</Typography>
            </Box>
          )}
        </Paper>

        {/* Alert Details Dialog */}
        {selectedAlertId && alertDetail && (
          <Dialog open={detailOpen} onClose={() => setDetailOpen(false)} maxWidth="sm" fullWidth>
            <DialogTitle>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                Alert Details
                <IconButton onClick={() => setDetailOpen(false)} size="small">
                  <CloseIcon />
                </IconButton>
              </Box>
            </DialogTitle>
            <DialogContent dividers>
              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <Typography variant="caption" color="textSecondary">
                    Detector
                  </Typography>
                  <Typography variant="body2">{(alertDetail as any)?.detector}</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="caption" color="textSecondary">
                    Severity
                  </Typography>
                  <Chip label={(alertDetail as any)?.severity} size="small" />
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="caption" color="textSecondary">
                    Status
                  </Typography>
                  <Chip
                    label={(alertDetail as any)?.acknowledged ? 'Acknowledged' : 'New'}
                    size="small"
                    color={(alertDetail as any)?.acknowledged ? 'success' : 'error'}
                  />
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="caption" color="textSecondary">
                    Source IP
                  </Typography>
                  <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                    {(alertDetail as any)?.src_ip}
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="caption" color="textSecondary">
                    Dest IP
                  </Typography>
                  <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                    {(alertDetail as any)?.dst_ip}
                  </Typography>
                </Grid>
                <Grid item xs={12}>
                  <Typography variant="caption" color="textSecondary">
                    Description
                  </Typography>
                  <Typography variant="body2">{(alertDetail as any)?.description}</Typography>
                </Grid>
              </Grid>
            </DialogContent>
            <DialogActions>
              <Button onClick={() => setDetailOpen(false)}>Close</Button>
              <Button
                variant="contained"
                onClick={() => {
                  handleAcknowledge(selectedAlertId);
                  setDetailOpen(false);
                }}
              >
                Acknowledge
              </Button>
            </DialogActions>
          </Dialog>
        )}

        {/* Filter Dialog */}
        <FilterDialog open={filterOpen} onClose={() => setFilterOpen(false)} type="alerts" />

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

export default AlertsPage;
