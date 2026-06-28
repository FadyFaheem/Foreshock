import {
  Box,
  Chip,
  Divider,
  LinearProgress,
  List,
  ListItem,
  ListItemText,
  Paper,
  Stack,
  Typography,
} from '@mui/material';
import type { Diagnosis } from '../api/foreshock';

type ChipColor = 'default' | 'success' | 'warning' | 'error' | 'info';

const SEV_COLOR: Record<string, ChipColor> = {
  none: 'success',
  low: 'info',
  medium: 'warning',
  high: 'error',
};

interface Props {
  diagnosis: Diagnosis | null;
  loading?: boolean;
}

export default function DiagnosisCard({ diagnosis, loading }: Props) {
  return (
    <Paper sx={{ p: 2, height: '100%' }}>
      <Stack direction="row" justifyContent="space-between" alignItems="center">
        <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
          RAG + LLM Diagnosis
        </Typography>
        {diagnosis && (
          <Chip
            size="small"
            variant="outlined"
            color={diagnosis.used_llm ? 'primary' : 'default'}
            label={diagnosis.used_llm ? 'LLM grounded' : 'Templated (LLM offline)'}
          />
        )}
      </Stack>

      {loading && <LinearProgress sx={{ my: 2 }} />}
      {!loading && !diagnosis && (
        <Typography color="text.secondary" sx={{ mt: 1 }}>
          Run a diagnosis to see a grounded result with sources.
        </Typography>
      )}

      {diagnosis && (
        <Box sx={{ mt: 1 }}>
          <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap" sx={{ mb: 1 }}>
            <Typography variant="h6">{diagnosis.label}</Typography>
            <Chip
              size="small"
              color={SEV_COLOR[diagnosis.severity] ?? 'default'}
              label={`severity: ${diagnosis.severity}`}
            />
            <Chip size="small" variant="outlined" label={`${(diagnosis.confidence * 100).toFixed(0)}% conf`} />
            <Chip size="small" variant="outlined" label={`${Math.round(diagnosis.rpm)} RPM`} />
          </Stack>

          <Typography variant="body2" sx={{ mb: 1 }}>
            {diagnosis.summary}
          </Typography>
          {diagnosis.likely_cause && (
            <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
              <strong>Likely cause:</strong> {diagnosis.likely_cause}
            </Typography>
          )}

          <Typography variant="body2" sx={{ fontWeight: 600, mt: 1 }}>
            Recommended actions
          </Typography>
          <List dense sx={{ py: 0 }}>
            {diagnosis.recommended_actions.map((a, i) => (
              <ListItem key={i} sx={{ py: 0 }}>
                <ListItemText primary={`- ${a}`} primaryTypographyProps={{ variant: 'body2' }} />
              </ListItem>
            ))}
          </List>

          <Divider sx={{ my: 1 }} />
          <Typography variant="body2" sx={{ fontWeight: 600 }}>
            Retrieved sources
          </Typography>
          <Stack spacing={0.5} sx={{ mt: 0.5 }}>
            {diagnosis.sources.map((s, i) => (
              <Box key={i} sx={{ display: 'flex', justifyContent: 'space-between', gap: 1 }}>
                <Typography variant="caption" color="text.secondary">
                  {s.title} ({s.source})
                </Typography>
                <Typography variant="caption">{s.score.toFixed(3)}</Typography>
              </Box>
            ))}
          </Stack>
        </Box>
      )}
    </Paper>
  );
}
