import React, { useState } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
  Box,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from '@mui/material';
import { useFilterStore } from '@stores/filterStore';

interface FilterDialogProps {
  open: boolean;
  onClose: () => void;
  type: 'flows' | 'alerts' | 'incidents';
}

export const FilterDialog: React.FC<FilterDialogProps> = ({ open, onClose, type }) => {
  const { flowFilters, alertFilters, incidentFilters, setFlowFilters, setAlertFilters, setIncidentFilters } = useFilterStore();
  const [localFilters, setLocalFilters] = useState<any>(
    type === 'flows' ? flowFilters : type === 'alerts' ? alertFilters : incidentFilters
  );

  const handleApply = () => {
    if (type === 'flows') setFlowFilters(localFilters);
    else if (type === 'alerts') setAlertFilters(localFilters);
    else setIncidentFilters(localFilters);
    onClose();
  };

  const handleReset = () => {
    const defaultFilters =
      type === 'flows'
        ? { page: 0, pageSize: 50, sortBy: 'start_time', sortOrder: 'desc' as const }
        : type === 'alerts'
        ? { page: 0, pageSize: 50, sortBy: 'timestamp', sortOrder: 'desc' as const }
        : { page: 0, pageSize: 50, sortBy: 'created_at', sortOrder: 'desc' as const };
    setLocalFilters(defaultFilters);
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle>Filter {type.charAt(0).toUpperCase() + type.slice(1)}</DialogTitle>
      <DialogContent sx={{ pt: 2 }}>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          {type === 'flows' && (
            <>
              <TextField
                label="Source IP"
                value={localFilters.sourceIp || ''}
                onChange={(e) => setLocalFilters({ ...localFilters, sourceIp: e.target.value })}
                size="small"
              />
              <TextField
                label="Destination IP"
                value={localFilters.destIp || ''}
                onChange={(e) => setLocalFilters({ ...localFilters, destIp: e.target.value })}
                size="small"
              />
              <TextField
                label="Protocol"
                value={localFilters.protocol || ''}
                onChange={(e) => setLocalFilters({ ...localFilters, protocol: e.target.value })}
                size="small"
              />
            </>
          )}
          {type === 'alerts' && (
            <>
              <FormControl size="small">
                <InputLabel>Severity</InputLabel>
                <Select
                  value={localFilters.severity || ''}
                  onChange={(e) => setLocalFilters({ ...localFilters, severity: e.target.value })}
                  label="Severity"
                >
                  <MenuItem value="low">Low</MenuItem>
                  <MenuItem value="medium">Medium</MenuItem>
                  <MenuItem value="high">High</MenuItem>
                  <MenuItem value="critical">Critical</MenuItem>
                </Select>
              </FormControl>
              <TextField
                label="Detector"
                value={localFilters.detector || ''}
                onChange={(e) => setLocalFilters({ ...localFilters, detector: e.target.value })}
                size="small"
              />
            </>
          )}
          {type === 'incidents' && (
            <>
              <FormControl size="small">
                <InputLabel>Status</InputLabel>
                <Select
                  value={localFilters.status || ''}
                  onChange={(e) => setLocalFilters({ ...localFilters, status: e.target.value })}
                  label="Status"
                >
                  <MenuItem value="open">Open</MenuItem>
                  <MenuItem value="investigating">Investigating</MenuItem>
                  <MenuItem value="resolved">Resolved</MenuItem>
                  <MenuItem value="closed">Closed</MenuItem>
                </Select>
              </FormControl>
              <TextField
                label="Assigned To"
                value={localFilters.assignedTo || ''}
                onChange={(e) => setLocalFilters({ ...localFilters, assignedTo: e.target.value })}
                size="small"
              />
            </>
          )}
        </Box>
      </DialogContent>
      <DialogActions>
        <Button onClick={handleReset} color="secondary">
          Reset
        </Button>
        <Button onClick={onClose}>Cancel</Button>
        <Button onClick={handleApply} variant="contained">
          Apply
        </Button>
      </DialogActions>
    </Dialog>
  );
};
