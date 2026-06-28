import { useState } from 'react';
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
import CasinoIcon from '@mui/icons-material/Casino';
import WaveformChart from './WaveformChart';
import { randomTest, type RandomTestResult, type Sample } from '../api/foreshock';
import { CONDITION_COLORS } from '../theme/theme';

interface Props {
  samples: Sample[];
}

export default function RandomTester({ samples }: Props) {
  const [condition, setCondition] = useState('random');
  const [noise, setNoise] = useState(0);
  const [result, setResult] = useState<RandomTestResult | null>(null);
  const [tally, setTally] = useState({ correct: 0, total: 0 });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onGenerate = async () => {
    setLoading(true);
    setError(null);
    try {
      const r = await randomTest(condition, noise);
      setResult(r);
      setTally((t) => ({
        correct: t.correct + (r.correct ? 1 : 0),
        total: t.total + 1,
      }));
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Test failed');
    } finally {
      setLoading(false);
    }
  };

  const acc = tally.total ? (100 * tally.correct) / tally.total : 0;
  const accColor = acc >= 80 ? 'success' : acc >= 50 ? 'warning' : 'error';

  return (
    <Paper sx={{ p: 2 }}>
      <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 1 }}>
        <Box>
          <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
            Random failure tester
          </Typography>
          <Typography variant="caption" color="text.secondary">
            Generate a random labeled window and see whether the model classifies it correctly.
          </Typography>
        </Box>
        {tally.total > 0 && (
          <Chip
            color={accColor}
            label={`Accuracy: ${tally.correct}/${tally.total} (${acc.toFixed(0)}%)`}
          />
        )}
      </Stack>

      <Stack
        direction={{ xs: 'column', sm: 'row' }}
        spacing={2}
        alignItems={{ sm: 'center' }}
        sx={{ mb: 1 }}
      >
        <FormControl size="small" sx={{ minWidth: 200 }}>
          <InputLabel id="rt-cond">Failure type</InputLabel>
          <Select
            labelId="rt-cond"
            label="Failure type"
            value={condition}
            onChange={(e: SelectChangeEvent) => setCondition(e.target.value)}
          >
            <MenuItem value="random">Random</MenuItem>
            {samples.map((s) => (
              <MenuItem key={s.id} value={s.id}>
                {s.label}
              </MenuItem>
            ))}
          </Select>
        </FormControl>

        <Box sx={{ minWidth: 170 }}>
          <Typography variant="caption" color="text.secondary">
            Noise (stress test): {(noise * 100).toFixed(0)}%
          </Typography>
          <Slider
            size="small"
            value={noise}
            min={0}
            max={1}
            step={0.05}
            onChange={(_, v) => setNoise(v as number)}
          />
        </Box>

        <Button
          variant="contained"
          startIcon={<CasinoIcon />}
          onClick={onGenerate}
          disabled={loading}
        >
          {loading ? 'Testing...' : 'Generate & test'}
        </Button>
        {tally.total > 0 && (
          <Button
            size="small"
            onClick={() => {
              setTally({ correct: 0, total: 0 });
              setResult(null);
            }}
          >
            Reset
          </Button>
        )}
      </Stack>

      {loading && <LinearProgress sx={{ my: 1 }} />}
      {error && (
        <Alert severity="error" sx={{ my: 1 }}>
          {error}
        </Alert>
      )}

      {result && (
        <Box>
          <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap" sx={{ mb: 1 }}>
            <Chip
              color={result.correct ? 'success' : 'error'}
              label={result.correct ? 'CORRECT' : 'WRONG'}
            />
            <Typography variant="body2">
              actual:{' '}
              <Box component="span" sx={{ fontWeight: 700, color: CONDITION_COLORS[result.actual] }}>
                {result.actual_label}
              </Box>
              {'  ->  predicted: '}
              <Box
                component="span"
                sx={{ fontWeight: 700, color: CONDITION_COLORS[result.prediction] }}
              >
                {result.prediction_label}
              </Box>
            </Typography>
            <Chip size="small" variant="outlined" label={`${(result.confidence * 100).toFixed(0)}% conf`} />
            {result.noise > 0 && (
              <Chip size="small" variant="outlined" label={`noise ${(result.noise * 100).toFixed(0)}%`} />
            )}
          </Stack>
          <WaveformChart t={result.waveform.t} x={result.waveform.x} />
        </Box>
      )}
    </Paper>
  );
}
