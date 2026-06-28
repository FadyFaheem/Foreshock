import { useEffect, useState } from 'react';
import {
  Alert,
  Box,
  Button,
  Chip,
  FormControl,
  InputLabel,
  LinearProgress,
  MenuItem,
  Paper,
  Select,
  Slider,
  Stack,
  Typography,
} from '@mui/material';
import type { SelectChangeEvent } from '@mui/material';
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
import AgentWorkflow from '../components/AgentWorkflow';
import DiagnosisCard from '../components/DiagnosisCard';
import SpectrumChart from '../components/SpectrumChart';
import {
  getAIStatus,
  getInjectBase,
  injectDiagnose,
  type AIStatus,
  type InjectBase,
  type InjectDiagnoseResult,
} from '../api/foreshock';
import { BRAND_COLORS } from '../theme/theme';

type ChartClick = { activeLabel?: string | number };

type AlertColor = 'success' | 'info' | 'warning' | 'error';

const SEV_ALERT: Record<string, AlertColor> = {
  none: 'success',
  low: 'info',
  medium: 'warning',
  high: 'error',
};

// "custom" = manually click impulses; the rest generate a realistic periodic
// fault at that bearing frequency, which the classifier actually recognizes.
const FAULT_OPTIONS = [
  { value: 'outer_race', label: 'Outer race (BPFO)' },
  { value: 'inner_race', label: 'Inner race (BPFI)' },
  { value: 'ball', label: 'Ball (BSF)' },
  { value: 'custom', label: 'Custom (click to place)' },
];

function msg(e: unknown): string {
  return e instanceof Error ? e.message : 'Request failed';
}

export default function FaultLabPage() {
  const [base, setBase] = useState<InjectBase | null>(null);
  const [faultType, setFaultType] = useState('outer_race');
  const [points, setPoints] = useState<number[]>([]);
  const [severity, setSeverity] = useState(1.5);
  const [result, setResult] = useState<InjectDiagnoseResult | null>(null);
  const [status, setStatus] = useState<AIStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const isCustom = faultType === 'custom';

  const loadBase = () => {
    setError(null);
    setResult(null);
    setPoints([]);
    getInjectBase()
      .then(setBase)
      .catch((e) => setError(msg(e)));
  };

  useEffect(() => {
    getInjectBase().then(setBase).catch((e) => setError(msg(e)));
    getAIStatus().then(setStatus).catch(() => undefined);
  }, []);

  const onFaultTypeChange = (e: SelectChangeEvent) => {
    setFaultType(e.target.value);
    setResult(null);
    setPoints([]);
  };

  const addPoint = (state: ChartClick) => {
    if (!isCustom || result || state?.activeLabel == null) return;
    const idx = Math.round(Number(state.activeLabel));
    setPoints((prev) => (prev.includes(idx) ? prev : [...prev, idx].slice(0, 30)));
  };

  const onAnalyze = async () => {
    if (!base) return;
    setLoading(true);
    setError(null);
    try {
      const req = isCustom
        ? { signal: base.signal, fs: base.fs, rpm: base.rpm, points, amplitude: severity }
        : { signal: base.signal, fs: base.fs, rpm: base.rpm, fault_type: faultType, severity };
      setResult(await injectDiagnose(req));
    } catch (e) {
      setError(msg(e));
    } finally {
      setLoading(false);
    }
  };

  const diagnosis = result?.agent.diagnosis ?? null;
  const chartData = result
    ? result.waveform.x.map((v, i) => ({ i, x: v }))
    : base
      ? base.signal.map((v, i) => ({ i, x: v }))
      : [];
  const markedFreq = result ? Object.entries(result.marked_frequency)[0] : undefined;
  const faultLabel = FAULT_OPTIONS.find((o) => o.value === faultType)?.label ?? faultType;
  const analyzeDisabled = !base || loading || !!result || (isCustom && points.length === 0);

  return (
    <Stack spacing={2}>
      <Paper sx={{ p: 2 }}>
        <Typography variant="h6" sx={{ fontWeight: 700, mb: 0.5 }}>
          Fault Lab
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 1.5 }}>
          Synthesize a bearing fault on a healthy signal and let the agent analyze it
          end to end. Generate a realistic fault of a chosen type (a periodic impulse
          train at its characteristic frequency, which the model is trained to catch),
          or switch to Custom to hand-place impulses. The AI points out the fault and
          where it shows up in the envelope spectrum.
        </Typography>

        <Stack
          direction={{ xs: 'column', sm: 'row' }}
          spacing={2}
          alignItems={{ sm: 'center' }}
        >
          <FormControl size="small" sx={{ minWidth: 220 }} disabled={loading || !!result}>
            <InputLabel id="fault-type">Fault to generate</InputLabel>
            <Select
              labelId="fault-type"
              label="Fault to generate"
              value={faultType}
              onChange={onFaultTypeChange}
            >
              {FAULT_OPTIONS.map((o) => (
                <MenuItem key={o.value} value={o.value}>
                  {o.label}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          <Box sx={{ minWidth: 200 }}>
            <Typography variant="caption" color="text.secondary">
              Severity: {severity.toFixed(1)}x
            </Typography>
            <Slider
              size="small"
              value={severity}
              min={0.2}
              max={3}
              step={0.1}
              disabled={!!result || loading}
              onChange={(_, v) => setSeverity(v as number)}
            />
          </Box>
          <Button variant="contained" onClick={onAnalyze} disabled={analyzeDisabled}>
            {loading
              ? 'Analyzing...'
              : isCustom
                ? `Inject ${points.length} defect${points.length === 1 ? '' : 's'} & analyze`
                : `Generate ${faultLabel} & analyze`}
          </Button>
          {(result || (isCustom && points.length > 0)) && (
            <Button size="small" onClick={loadBase} disabled={loading}>
              {result ? 'Reset' : 'Clear points'}
            </Button>
          )}
          {status && (
            <Stack direction="row" spacing={1} sx={{ ml: { sm: 'auto' } }}>
              <Chip
                size="small"
                variant="outlined"
                color={status.llm ? 'success' : 'default'}
                label={status.llm ? `LLM: ${status.model}` : 'LLM offline'}
              />
              <Chip
                size="small"
                variant="outlined"
                color={status.db ? 'success' : 'default'}
                label={`KB: ${status.kb_size}`}
              />
            </Stack>
          )}
        </Stack>

        {status && !status.llm && (
          <Alert severity="warning" sx={{ mt: 1.5 }}>
            LLM offline &mdash; the diagnosis falls back to templated text. Start Ollama
            and pull the model for a grounded analysis.
          </Alert>
        )}
      </Paper>

      {loading && <LinearProgress />}
      {error && <Alert severity="error">{error}</Alert>}

      {diagnosis && (
        <Alert
          severity={SEV_ALERT[diagnosis.severity] ?? 'info'}
          icon={false}
          sx={{ '& .MuiAlert-message': { width: '100%' } }}
        >
          <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap">
            <Typography variant="subtitle1" sx={{ fontWeight: 700 }}>
              AI pinpointed: {diagnosis.label}
            </Typography>
            <Chip size="small" label={`${(diagnosis.confidence * 100).toFixed(0)}% confidence`} />
            <Chip size="small" label={`severity: ${diagnosis.severity}`} />
            {markedFreq && (
              <Chip
                size="small"
                color="error"
                variant="outlined"
                label={`evidence: ${markedFreq[0]} ≈ ${markedFreq[1]} Hz`}
              />
            )}
            {!markedFreq && diagnosis.health?.caught && (
              <Chip
                size="small"
                color="error"
                variant="outlined"
                label={`anomaly: error ${diagnosis.health.error.toFixed(2)} > ${diagnosis.health.threshold.toFixed(2)}`}
              />
            )}
          </Stack>
        </Alert>
      )}

      <Paper sx={{ p: 2 }}>
        <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>
          {result ? 'Synthesized signal (with injected fault)' : 'Healthy signal'}
        </Typography>
        <Box sx={{ width: '100%', height: 240 }}>
          <ResponsiveContainer>
            <LineChart
              data={chartData}
              margin={{ top: 8, right: 16, bottom: 8, left: 0 }}
              onClick={addPoint}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="i"
                type="number"
                domain={['dataMin', 'dataMax']}
                tick={{ fontSize: 11 }}
              />
              <YAxis tick={{ fontSize: 11 }} width={48} />
              <Tooltip
                formatter={(v) => Number(v).toFixed(3)}
                labelFormatter={(l) => `sample ${l}`}
              />
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
            {isCustom
              ? 'Tip: click a few spots to place defects, set severity, then analyze. Isolated impulses are hard to classify - generate a typed fault for a clean detection.'
              : `Generates a realistic ${faultLabel} fault: a periodic impulse train at its characteristic frequency. Raise severity for a stronger fault.`}
          </Typography>
        )}
      </Paper>

      {result && (
        <SpectrumChart
          title="Envelope spectrum — where the AI sees the fault"
          f={result.envelope.f}
          mag={result.envelope.mag}
          color={BRAND_COLORS.primary}
          faultFrequencies={result.marked_frequency}
        />
      )}

      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' }, gap: 2 }}>
        <DiagnosisCard diagnosis={diagnosis} loading={loading} />
        <AgentWorkflow run={result?.agent ?? null} loading={loading} />
      </Box>
    </Stack>
  );
}
