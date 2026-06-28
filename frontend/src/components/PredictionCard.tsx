import { Box, Chip, LinearProgress, Paper, Stack, Typography } from '@mui/material';
import type { PredictionResponse } from '../api/foreshock';
import { CONDITION_COLORS } from '../theme/theme';

interface Props {
  prediction: PredictionResponse | null;
  loading?: boolean;
}

export default function PredictionCard({ prediction, loading }: Props) {
  return (
    <Paper sx={{ p: 2, height: '100%' }}>
      <Typography variant="overline" color="text.secondary">
        Prediction
      </Typography>

      {loading && <LinearProgress sx={{ my: 2 }} />}

      {!loading && !prediction && (
        <Typography color="text.secondary" sx={{ mt: 1 }}>
          Select a sample or upload a file to see a prediction.
        </Typography>
      )}

      {prediction && (
        <>
          <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 0.5 }}>
            <Typography
              variant="h5"
              sx={{
                fontWeight: 700,
                color: CONDITION_COLORS[prediction.prediction] ?? 'text.primary',
              }}
            >
              {prediction.prediction_label}
            </Typography>
            <Chip
              size="small"
              color="primary"
              label={`${(prediction.confidence * 100).toFixed(1)}%`}
            />
          </Stack>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Averaged over {prediction.n_windows} window
            {prediction.n_windows === 1 ? '' : 's'} · {Math.round(prediction.rpm)} RPM
          </Typography>

          <Stack spacing={1.25}>
            {prediction.probabilities.map((p) => (
              <Box key={p.condition}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                  <Typography
                    variant="body2"
                    sx={{
                      fontWeight: p.condition === prediction.prediction ? 700 : 400,
                    }}
                  >
                    {p.label}
                  </Typography>
                  <Typography variant="body2">
                    {(p.probability * 100).toFixed(1)}%
                  </Typography>
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={Math.min(100, p.probability * 100)}
                  sx={{
                    height: 8,
                    borderRadius: 1,
                    bgcolor: 'grey.200',
                    '& .MuiLinearProgress-bar': {
                      backgroundColor: CONDITION_COLORS[p.condition] ?? 'primary.main',
                    },
                  }}
                />
              </Box>
            ))}
          </Stack>
        </>
      )}
    </Paper>
  );
}
