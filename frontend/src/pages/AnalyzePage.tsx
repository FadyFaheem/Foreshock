import { useCallback, useEffect, useState } from 'react';
import { Alert, Box, Paper, Stack, Typography } from '@mui/material';
import RandomTester from '../components/RandomTester';
import SignalPicker from '../components/SignalPicker';
import WaveformChart from '../components/WaveformChart';
import SpectrumChart from '../components/SpectrumChart';
import PredictionCard from '../components/PredictionCard';
import FeatureTable from '../components/FeatureTable';
import { BRAND_COLORS } from '../theme/theme';
import {
  getSamples,
  getSignal,
  predictFile,
  predictSample,
  type PredictionResponse,
  type Sample,
  type SignalResponse,
} from '../api/foreshock';

export default function AnalyzePage() {
  const [samples, setSamples] = useState<Sample[]>([]);
  const [selectedId, setSelectedId] = useState('');
  const [signal, setSignal] = useState<SignalResponse | null>(null);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadSample = useCallback(async (id: string) => {
    setLoading(true);
    setError(null);
    try {
      const [sig, pred] = await Promise.all([getSignal(id), predictSample(id)]);
      setSignal(sig);
      setPrediction(pred);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load sample');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    getSamples()
      .then((list) => {
        setSamples(list);
        if (list.length > 0) {
          setSelectedId(list[0].id);
          void loadSample(list[0].id);
        }
      })
      .catch((e) =>
        setError(e instanceof Error ? e.message : 'Failed to load samples'),
      );
  }, [loadSample]);

  const handleSelect = (id: string) => {
    setSelectedId(id);
    void loadSample(id);
  };

  const handleUpload = async (file: File) => {
    setLoading(true);
    setError(null);
    try {
      const pred = await predictFile(file);
      setPrediction(pred);
      setSignal(null); // there is no waveform endpoint for arbitrary uploads
      setSelectedId('');
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to analyze file');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Stack spacing={2}>
      <SignalPicker
        samples={samples}
        selectedId={selectedId}
        onSelect={handleSelect}
        onUpload={handleUpload}
        loading={loading}
      />

      {error && <Alert severity="error">{error}</Alert>}

      {signal ? (
        <>
          <WaveformChart t={signal.waveform.t} x={signal.waveform.x} />
          <Box
            sx={{
              display: 'grid',
              gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' },
              gap: 2,
            }}
          >
            <SpectrumChart
              title="FFT spectrum"
              f={signal.spectrum.f}
              mag={signal.spectrum.mag}
              color={BRAND_COLORS.primary}
            />
            <SpectrumChart
              title="Envelope spectrum (Hilbert)"
              f={signal.envelope.f}
              mag={signal.envelope.mag}
              color={BRAND_COLORS.secondary}
              faultFrequencies={signal.fault_frequencies}
            />
          </Box>
        </>
      ) : (
        prediction && (
          <Paper sx={{ p: 2 }}>
            <Typography color="text.secondary">
              Waveform and spectrum previews are shown for the built-in samples.
              Your uploaded signal was analyzed; results are below.
            </Typography>
          </Paper>
        )
      )}

      <Box
        sx={{
          display: 'grid',
          gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' },
          gap: 2,
        }}
      >
        <PredictionCard prediction={prediction} loading={loading} />
        <FeatureTable features={prediction?.features ?? []} />
      </Box>

      <RandomTester samples={samples} />
    </Stack>
  );
}
