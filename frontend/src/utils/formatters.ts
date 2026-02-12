// Format validation errors
export const formatError = (error: any): string => {
  if (typeof error === 'string') return error;
  if (error?.message) return error.message;
  if (error?.detail) return error.detail;
  return 'An unexpected error occurred';
};

// Format bytes to human-readable
export const formatBytes = (bytes: number): string => {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

// Format large numbers with commas
export const formatNumber = (num: number): string => {
  return num.toLocaleString();
};

// Format IP address
export const formatIp = (ip: string): string => {
  return ip || 'N/A';
};

// Get risk level color
export const getRiskColor = (riskScore: number): string => {
  if (riskScore >= 80) return '#d32f2f'; // Critical - Red
  if (riskScore >= 60) return '#f57c00'; // High - Orange
  if (riskScore >= 40) return '#fbc02d'; // Medium - Yellow
  return '#388e3c'; // Low - Green
};

// Get severity color
export const getSeverityColor = (severity: string): string => {
  switch (severity.toLowerCase()) {
    case 'critical':
      return '#d32f2f';
    case 'high':
      return '#f57c00';
    case 'medium':
      return '#fbc02d';
    case 'low':
      return '#388e3c';
    default:
      return '#666666';
  }
};

// Format timestamp
export const formatTimestamp = (timestamp: string): string => {
  try {
    return new Date(timestamp).toLocaleString();
  } catch {
    return timestamp;
  }
};
