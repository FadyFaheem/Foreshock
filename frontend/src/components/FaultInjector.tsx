import { useEffect, useState } from 'react';
import {
  Alert,
  Box,
  Button,
  Chip,
  LinearProgress,
  Slider,
  Stack,
  Typography,
} from '@mui/material';
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import {
  getInjectBase,
  injectFaults,
  type InjectBase,
  type InjectResult,
} from '../api/foreshock';
import { BRAND_COLORS } from '../theme/theme';

type ChartClick = { activeLabel?: string | number };

export default function FaultInjector() {
  const [base, setBase] = useState<InjectBase | null>(null);
  const [points, setPoints] = useState<number[]>([]);
  const [amplitude, setAmplitude] = useState(1.0);
  const [result, setResult] = useState<InjectResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadBase = () => {
    setError(null);
    setResult(null);
    setPoints([]);
    getInjectBase()
      .then(setBase)
      .catch((e) => setError(e instanceof Error ? e.message : 'Failed to load signal'));
  };

  useEffect(() => {
    getInjectBase()
      .then(setBase)
      .catch((e) => setError(e instanceof Error ? e.message : 'Failed to load signal'));
  }, []);

  const chartData = result
    ? result.waveform.x.map((v, i) => ({ i, x: v }))
    : base
      ? base.signal.map((v, i) => ({ i, x: v }))
      : [];

  const addPoint = (state: ChartClick) => {
    if (result || state?.activeLabel == null) return; // locked after detect
    const idx = Math.round(Number(state.activeLabel));
    setPoints((prev) => (prev.includes(idx) ? prev : [...prev, idx].slice(0, 30)));
  };

  const onDetect = async () => {
    if (!base || points.length === 0) return;
    setLoading(true);
    setError(null);
    try {
      setResult(await injectFaults(base.signal, points, amplitude, base.fs, base.rpm));
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Detection failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 1.5 }}>
        Start from a healthy signal, click the waveform to place defects at specific spots,
        set their severity, then detect. "Caught" means the classifier or the health monitor
        flagged the signal as not healthy.
      </Typography>

      <Stack
        direction={{ xs: 'column', sm: 'row' }}
        spacing={2}
        alignItems={{ sm: 'center' }}
        sx={{ mb: 1.5 }}
      >
        <Button size="small" variant="outlined" onClick={loadBase}>
          New healthy signal
        </Button>
        <Box sx={{ minWidth: 190 }}>
          <Typography variant="caption" color="text.secondary">
            Defect severity: {amplitude.toFixed(1)}x
          </Typography>
          <Slider
            size="small"
            value={amplitude}
            min={0.2}
            max={3}
            step={0.1}
            disabled={!!result}
            onChange={(_, v) => setAmplitude(v as number)}
          />
        </Box>
        <Button
          variant="contained"
          onClick={onDetect}
          disabled={!base || points.length === 0 || loading || !!result}
        >
          Inject {points.length} defect{points.length === 1 ? '' : 's'} &amp; detect
        </Button>
        {result && (
          <Button size="small" onClick={loadBase}>
            Reset
          </Button>
        )}
      </Stack>

      {loading && <LinearProgress sx={{ mb: 1 }} />}
      {error && (
        <Alert severity="error" sx={{ mb: 1 }}>
          {error}
        </Alert>
      )}

      {result && (
        <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap" sx={{ mb: 1 }}>
          <Chip color={result.caught ? 'success' : 'error'} label={result.caught ? 'CAUGHT' : 'MISSED'} />
          <Typography variant="body2">
            classifier: <strong>{result.prediction_label}</strong> ({(result.confidence * 100).toFixed(0)}%)
          </Typography>
          {result.health && (
            <Chip
              size="small"
              variant="outlined"
              color={result.health.caught ? 'success' : 'default'}
              label={`health error ${result.health.error.toFixed(2)} / thr ${result.health.threshold.toFixed(2)}`}
            />
          )}
        </Stack>
      )}

      <Box sx={{ width: '100%', height: 240 }}>
        <ResponsiveContainer>
          <LineChart
            data={chartData}
            margin={{ top: 8, right: 16, bottom: 8, left: 0 }}
            onClick={addPoint}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="i" type="number" domain={['dataMin', 'dataMax']} tick={{ fontSize: 11 }} />
            <YAxis tick={{ fontSize: 11 }} width={48} />
            <Tooltip formatter={(v) => Number(v).toFixed(3)} labelFormatter={(l) => `sample ${l}`} />
            <Line
              type="monotone"
              dataKey="x"
              stroke={result ? BRAND_COLORS.secondary : BRAND_COLORS.primary}
              dot={false}
              strokeWidth={1}
              isAnimationActive={false}
            />
            {points.map((p) => (
              <ReferenceLine key={p} x={p} stroke="#d32f2f" strokeDasharray="2 2" />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </Box>
      {!result && (
        <Typography variant="caption" color="text.secondary">
          Tip: click a few spots on the waveform, then inject. Higher severity is easier to catch.
        </Typography>
      )}
    </Box>
  );
}
