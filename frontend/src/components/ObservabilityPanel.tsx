import {
  Box,
  Button,
  Paper,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Typography,
} from '@mui/material';
import type { Observability } from '../api/foreshock';

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <Box sx={{ textAlign: 'center', px: 2, py: 0.5 }}>
      <Typography variant="h6">{value}</Typography>
      <Typography variant="caption" color="text.secondary">
        {label}
      </Typography>
    </Box>
  );
}

interface Props {
  data: Observability | null;
  onRefresh: () => void;
}

export default function ObservabilityPanel({ data, onRefresh }: Props) {
  return (
    <Paper sx={{ p: 2 }}>
      <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 1 }}>
        <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
          AI Observability
        </Typography>
        <Button size="small" onClick={onRefresh}>
          Refresh
        </Button>
      </Stack>

      {!data && <Typography color="text.secondary">No call data yet.</Typography>}

      {data && (
        <Box>
          <Stack direction="row" flexWrap="wrap" sx={{ mb: 1 }}>
            <Stat label="LLM/embed calls" value={String(data.summary.calls)} />
            <Stat label="avg latency" value={`${Math.round(data.summary.avg_latency_ms)} ms`} />
            <Stat label="total tokens" value={String(data.summary.total_tokens)} />
            <Stat label="avg retrieval" value={data.summary.avg_retrieval_score.toFixed(2)} />
            <Stat label="errors" value={String(data.summary.errors)} />
          </Stack>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>operation</TableCell>
                <TableCell align="right">latency (ms)</TableCell>
                <TableCell align="right">tokens</TableCell>
                <TableCell align="right">retrieval</TableCell>
                <TableCell align="right">time</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {data.recent.slice(0, 8).map((c, i) => (
                <TableRow key={i}>
                  <TableCell>{c.operation}</TableCell>
                  <TableCell align="right">{Math.round(c.latency_ms)}</TableCell>
                  <TableCell align="right">{c.total_tokens ?? ''}</TableCell>
                  <TableCell align="right">
                    {c.retrieval_score != null ? c.retrieval_score.toFixed(2) : ''}
                  </TableCell>
                  <TableCell align="right">
                    {new Date(c.created_at).toLocaleTimeString()}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </Box>
      )}
    </Paper>
  );
}
