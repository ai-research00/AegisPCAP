import React from 'react';
import { Snackbar, Alert as MuiAlert, AlertProps } from '@mui/material';

interface ToastProps extends AlertProps {
  message: string;
  open: boolean;
  onClose: () => void;
  autoHideDuration?: number;
}

export const Toast: React.FC<ToastProps> = ({
  message,
  open,
  onClose,
  autoHideDuration = 6000,
  severity = 'info',
  ...props
}) => {
  return (
    <Snackbar
      open={open}
      autoHideDuration={autoHideDuration}
      onClose={onClose}
      anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
    >
      <MuiAlert onClose={onClose} severity={severity} sx={{ width: '100%' }} {...props}>
        {message}
      </MuiAlert>
    </Snackbar>
  );
};

// Toast container/state manager
export const useToast = () => {
  const [toasts, setToasts] = React.useState<Array<{ id: string; message: string; severity: AlertProps['severity'] }>>([]);

  const showToast = (message: string, severity: AlertProps['severity'] = 'info') => {
    const id = Date.now().toString();
    setToasts((prev) => [...prev, { id, message, severity }]);
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 6000);
  };

  return { toasts, showToast };
};
