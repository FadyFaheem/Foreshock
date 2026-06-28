import { useEffect, useState } from 'react';
import {
  Box,
  Dialog,
  DialogContent,
  DialogTitle,
  IconButton,
  Tab,
  Tabs,
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import FaultInjector from './FaultInjector';
import RandomTester from './RandomTester';
import { getSamples, type Sample } from '../api/foreshock';

interface Props {
  open: boolean;
  onClose: () => void;
}

export default function TesterDialog({ open, onClose }: Props) {
  const [tab, setTab] = useState(0);
  const [samples, setSamples] = useState<Sample[]>([]);

  useEffect(() => {
    if (open && samples.length === 0) {
      getSamples().then(setSamples).catch(() => undefined);
    }
  }, [open, samples.length]);

  return (
    <Dialog open={open} onClose={onClose} fullWidth maxWidth="md">
      <DialogTitle sx={{ pr: 6 }}>
        Signal tester
        <IconButton onClick={onClose} sx={{ position: 'absolute', right: 8, top: 8 }}>
          <CloseIcon />
        </IconButton>
      </DialogTitle>
      <Tabs value={tab} onChange={(_, v) => setTab(v)} sx={{ px: 3 }}>
        <Tab label="Random test" />
        <Tab label="Fault injection" />
      </Tabs>
      <DialogContent dividers>
        {tab === 0 && (
          <Box>
            <RandomTester samples={samples} />
          </Box>
        )}
        {tab === 1 && (
          <Box>
            <FaultInjector />
          </Box>
        )}
      </DialogContent>
    </Dialog>
  );
}
