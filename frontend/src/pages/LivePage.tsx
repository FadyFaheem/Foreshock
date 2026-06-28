import { useEffect, useRef, useState } from 'react';
import {
  Alert,
  Box,
  Button,
  Chip,
  Paper,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Typography,
} from '@mui/material';
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import {
  getStreamStatus,
  simulateStream,
  stopStream,
  type StreamEvent,
  type StreamStatus,
} from '../api/foreshock';
import { BRAND_COLORS, CONDITION_COLORS } from '../theme/theme';

export default function LivePage() {
  const [status, setStatus] = useState<StreamStatus | null>(null);
  const [events, setEvents] = useState<StreamEvent[]>([]);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const esRef = useRef<EventSource | null>(null);

  useEffect(() => {
    getStreamStatus().then(setStatus).catch(() => undefined);
    const es = new EventSource('/api/stream');
    esRef.current = es;
    es.onopen = () => setConnected(true);
    es.onerror = () => setConnected(false);
    es.addEventListener('hello', (e) => {
      try {
        const d = JSON.parse((e as MessageEvent).data);
        if (Array.isArray(d.recent)) setEvents(d.recent.slice().reverse());
      } catch {
        /* ignore */
      }
    });
    es.onmessage = (e) => {
      try {
        const ev = JSON.parse(e.data) as StreamEvent;
        setEvents((prev) => [ev, ...prev].slice(0, 50));
      } catch {
        /* ignore */
      }
    };
    return () => es.close();
  }, []);

  const onStart = async () => {
    setError(null);
    try {
      await simulateStream(24, 0.7);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to start feed');
    }
  };

  const onStop = async () => {
    try {
      await stopStream();
    } catch {
      /* ignore */
    }
  };

  const scored = events.filter((e) => e.correct !== null);
  const correct = scored.filter((e) => e.correct).length;
  const acc = scored.length ? (100 * correct) / scored.length : 0;
  const chartData = events
    .slice()
    .reverse()
    .map((e, i) => ({ i, confidence: +(e.confidence * 100).toFixed(1) }));

  return (
    <Stack spacing={2}>
      <Paper sx={{ p: 2 }}>
        <Stack direction="row" justifyContent="space-between" alignItems="center" flexWrap="wrap">
          <Box>
            <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
              Live sensor feed - Kafka
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Windows stream through Kafka; a consumer runs inference and pushes results here over SSE.
            </Typography>
          </Box>
          <Stack direction="row" spacing={1} alignItems="center">
            <Chip size="small" variant="outlined" color={connected ? 'success' : 'default'} label={connected ? 'SSE connected' : 'SSE...'} />
            <Chip size="small" variant="outlined" color={status?.kafka ? 'success' : 'default'} label={status?.kafka ? 'Kafka up' : 'Kafka down'} />
          </Stack>
        </Stack>
        <Stack direction="row" spacing={2} alignItems="center" sx={{ mt: 1.5 }} flexWrap="wrap">
          <Button variant="contained" onClick={onStart} disabled={!status?.kafka}>
            Start live feed
          </Button>
          <Button variant="outlined" onClick={onStop}>
            Stop
          </Button>
          {scored.length > 0 && (
            <Chip
              color={acc >= 80 ? 'success' : acc >= 50 ? 'warning' : 'error'}
              label={`Live accuracy: ${correct}/${scored.length} (${acc.toFixed(0)}%)`}
            />
          )}
        </Stack>
        {status && !status.kafka && (
          <Alert severity="warning" sx={{ mt: 1.5 }}>
            Kafka broker not reachable. Start the broker (it runs in the Podman pod) to stream.
          </Alert>
        )}
        {error && (
          <Alert severity="error" sx={{ mt: 1.5 }}>
            {error}
          </Alert>
        )}
      </Paper>

      <Paper sx={{ p: 2 }}>
        <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>
          Live confidence
        </Typography>
        <Box sx={{ width: '100%', height: 200 }}>
          <ResponsiveContainer>
            <LineChart data={chartData} margin={{ top: 5, right: 16, bottom: 5, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="i" tick={{ fontSize: 12 }} />
              <YAxis domain={[0, 100]} tick={{ fontSize: 12 }} width={40} />
              <Tooltip formatter={(v) => `${Number(v).toFixed(0)}%`} />
              <Line type="monotone" dataKey="confidence" stroke={BRAND_COLORS.primary} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </Box>
      </Paper>

      <Paper sx={{ p: 2 }}>
        <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>
          Incoming windows
        </Typography>
        {events.length === 0 ? (
          <Typography color="text.secondary">No events yet. Click "Start live feed".</Typography>
        ) : (
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>time</TableCell>
                <TableCell>actual</TableCell>
                <TableCell>predicted</TableCell>
                <TableCell align="right">conf</TableCell>
                <TableCell align="center">ok</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {events.slice(0, 12).map((e, i) => (
                <TableRow key={i}>
                  <TableCell>{new Date(e.ts * 1000).toLocaleTimeString()}</TableCell>
                  <TableCell>
                    <Box component="span" sx={{ color: e.actual ? CONDITION_COLORS[e.actual] : 'inherit' }}>
                      {e.actual_label ?? '-'}
                    </Box>
                  </TableCell>
                  <TableCell>
                    <Box component="span" sx={{ color: CONDITION_COLORS[e.prediction] }}>
                      {e.prediction_label}
                    </Box>
                  </TableCell>
                  <TableCell align="right">{(e.confidence * 100).toFixed(0)}%</TableCell>
                  <TableCell align="center">{e.correct == null ? '-' : e.correct ? 'yes' : 'no'}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </Paper>
    </Stack>
  );
}
