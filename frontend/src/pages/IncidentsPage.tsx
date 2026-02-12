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
  DialogActions,
  IconButton,
  Tooltip,
  Grid,
  Typography,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
} from '@mui/material';
import { FilterList as FilterListIcon, Close as CloseIcon, Edit as EditIcon } from '@mui/icons-material';
import { useIncidents, useIncidentDetails } from '@hooks/useApi';
import { useRealtimeIncidents } from '@hooks/useRealtime';
import { useFilterStore } from '@stores/filterStore';
import { ErrorBoundary, Loading, ErrorMessage } from '@components/ErrorBoundary';
import { FilterDialog } from '@components/FilterDialog';
import { Toast, useToast } from '@components/Toast';
import { getSeverityColor, formatTimestamp } from '@utils/formatters';

const IncidentsPage: React.FC = () => {
  const { incidentFilters, setIncidentFilters } = useFilterStore();
  const { data: incidentsData, isLoading, isError } = useIncidents(incidentFilters);
  const [filterOpen, setFilterOpen] = useState(false);
  const [detailOpen, setDetailOpen] = useState(false);
  const [editOpen, setEditOpen] = useState(false);

  // Enable real-time incident synchronization
  useRealtimeIncidents();
  const [selectedIncidentId, setSelectedIncidentId] = useState<string | null>(null);
  const { data: incidentDetail } = useIncidentDetails(selectedIncidentId || '');
  const [assignedTo, setAssignedTo] = useState('');
  const [newStatus, setNewStatus] = useState('');
  const { toasts, showToast } = useToast();

  if (isLoading) return <Loading />;
  if (isError) return <ErrorMessage message="Failed to load incidents" />;

  const incidents = (incidentsData as any)?.incidents || [];
  const totalCount = (incidentsData as any)?.total || 0;

  const handleViewDetails = (incidentId: string) => {
    setSelectedIncidentId(incidentId);
    setDetailOpen(true);
  };

  const handleEdit = (incidentId: string) => {
    setSelectedIncidentId(incidentId);
    setEditOpen(true);
  };

  const handleStatusUpdate = async () => {
    try {
      // API call would go here
      showToast('Incident status updated', 'success');
      setEditOpen(false);
    } catch (error) {
      showToast('Failed to update incident', 'error');
    }
  };

  const handleAssign = async () => {
    try {
      // API call would go here
      showToast('Incident assigned', 'success');
      setEditOpen(false);
    } catch (error) {
      showToast('Failed to assign incident', 'error');
    }
  };

  const getStatusColor = (status: string) => {
    const colors: Record<string, string> = {
      open: '#f44336',
      investigating: '#ff9800',
      resolved: '#4caf50',
      closed: '#9e9e9e',
    };
    return colors[status] || '#2196f3';
  };

  return (
    <ErrorBoundary>
      <Box sx={{ p: 3 }}>
        <Typography variant="h4" sx={{ mb: 3, fontWeight: 600 }}>
          Security Incidents
        </Typography>

        {/* Filters */}
        <Box sx={{ display: 'flex', gap: 2, mb: 3, alignItems: 'center' }}>
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
            Showing {incidents.length} of {totalCount} incidents
          </Typography>
        </Box>

        {/* Incidents Table */}
        <Paper>
          <TableContainer>
            <Table size="small">
              <TableHead>
                <TableRow sx={{ backgroundColor: '#f5f5f5' }}>
                  <TableCell sx={{ fontWeight: 600 }}>Title</TableCell>
                  <TableCell sx={{ fontWeight: 600 }}>Status</TableCell>
                  <TableCell sx={{ fontWeight: 600 }}>Severity</TableCell>
                  <TableCell sx={{ fontWeight: 600 }}>Assigned To</TableCell>
                  <TableCell sx={{ fontWeight: 600 }}>Created</TableCell>
                  <TableCell sx={{ fontWeight: 600 }} align="center">
                    Actions
                  </TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {incidents.map((incident: any, idx: number) => (
                  <TableRow key={idx} hover>
                    <TableCell>
                      <Typography
                        variant="body2"
                        sx={{ cursor: 'pointer', color: 'primary.main', '&:hover': { textDecoration: 'underline' } }}
                        onClick={() => handleViewDetails(incident.id)}
                      >
                        {incident.title}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={incident.status?.toUpperCase()}
                        size="small"
                        sx={{ backgroundColor: getStatusColor(incident.status || 'open'), color: 'white' }}
                      />
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={incident.severity?.toUpperCase()}
                        size="small"
                        sx={{ backgroundColor: getSeverityColor(incident.severity || 'medium'), color: 'white' }}
                      />
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">{incident.assigned_to || '-'}</Typography>
                    </TableCell>
                    <TableCell>
                      <Typography variant="caption">{formatTimestamp(incident.created_at)}</Typography>
                    </TableCell>
                    <TableCell align="center">
                      <Tooltip title="Edit">
                        <IconButton size="small" onClick={() => handleEdit(incident.id)}>
                          <EditIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
          <TablePagination
            rowsPerPageOptions={[10, 25, 50]}
            component="div"
            count={totalCount}
            rowsPerPage={incidentFilters.pageSize}
            page={incidentFilters.page}
            onPageChange={(e, newPage) => setIncidentFilters({ ...incidentFilters, page: newPage })}
            onRowsPerPageChange={(e) =>
              setIncidentFilters({ ...incidentFilters, pageSize: parseInt(e.target.value, 10), page: 0 })
            }
          />
        </Paper>

        {/* Incident Details Dialog */}
        {selectedIncidentId && incidentDetail && (
          <Dialog open={detailOpen} onClose={() => setDetailOpen(false)} maxWidth="md" fullWidth>
            <DialogTitle>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                Incident Details
                <IconButton onClick={() => setDetailOpen(false)} size="small">
                  <CloseIcon />
                </IconButton>
              </Box>
            </DialogTitle>
            <DialogContent dividers>
              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <Typography variant="caption" color="textSecondary">
                    Title
                  </Typography>
                  <Typography variant="h6">{(incidentDetail as any)?.title}</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="caption" color="textSecondary">
                    Status
                  </Typography>
                  <Chip label={(incidentDetail as any)?.status} size="small" />
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="caption" color="textSecondary">
                    Severity
                  </Typography>
                  <Chip label={(incidentDetail as any)?.severity} size="small" />
                </Grid>
                <Grid item xs={12}>
                  <Typography variant="caption" color="textSecondary">
                    Description
                  </Typography>
                  <Typography variant="body2">{(incidentDetail as any)?.description}</Typography>
                </Grid>
                <Grid item xs={12}>
                  <Typography variant="caption" color="textSecondary">
                    Assigned To
                  </Typography>
                  <Typography variant="body2">{(incidentDetail as any)?.assigned_to || 'Unassigned'}</Typography>
                </Grid>
              </Grid>
            </DialogContent>
            <DialogActions>
              <Button onClick={() => setDetailOpen(false)}>Close</Button>
              <Button variant="contained" onClick={() => { setDetailOpen(false); setEditOpen(true); }}>
                Edit
              </Button>
            </DialogActions>
          </Dialog>
        )}

        {/* Edit Incident Dialog */}
        {selectedIncidentId && (
          <Dialog open={editOpen} onClose={() => setEditOpen(false)} maxWidth="sm" fullWidth>
            <DialogTitle>Edit Incident</DialogTitle>
            <DialogContent sx={{ pt: 2 }}>
              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <FormControl fullWidth size="small">
                    <InputLabel>Status</InputLabel>
                    <Select
                      value={newStatus}
                      onChange={(e) => setNewStatus(e.target.value)}
                      label="Status"
                    >
                      <MenuItem value="open">Open</MenuItem>
                      <MenuItem value="investigating">Investigating</MenuItem>
                      <MenuItem value="resolved">Resolved</MenuItem>
                      <MenuItem value="closed">Closed</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={12}>
                  <FormControl fullWidth size="small">
                    <InputLabel>Assign To</InputLabel>
                    <Select
                      value={assignedTo}
                      onChange={(e) => setAssignedTo(e.target.value)}
                      label="Assign To"
                    >
                      <MenuItem value="">Unassigned</MenuItem>
                      <MenuItem value="analyst1">Analyst 1</MenuItem>
                      <MenuItem value="analyst2">Analyst 2</MenuItem>
                      <MenuItem value="analyst3">Analyst 3</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
              </Grid>
            </DialogContent>
            <DialogActions>
              <Button onClick={() => setEditOpen(false)}>Cancel</Button>
              <Button
                variant="contained"
                onClick={() => {
                  if (newStatus) handleStatusUpdate();
                  if (assignedTo) handleAssign();
                }}
              >
                Save Changes
              </Button>
            </DialogActions>
          </Dialog>
        )}

        {/* Filter Dialog */}
        <FilterDialog open={filterOpen} onClose={() => setFilterOpen(false)} type="incidents" />

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

export default IncidentsPage;
