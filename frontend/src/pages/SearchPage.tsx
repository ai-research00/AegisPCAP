import React from 'react';
import { Box, Typography, Paper } from '@mui/material';

const SearchPage: React.FC = () => {
  return (
    <Box>
      <Typography variant="h4" sx={{ mb: 3, fontWeight: 600 }}>
        Advanced Search
      </Typography>

      <Paper sx={{ p: 2 }}>
        <Typography color="textSecondary">
          Search page will be implemented in Phase 6.2b
        </Typography>
      </Paper>
    </Box>
  );
};

export default SearchPage;
