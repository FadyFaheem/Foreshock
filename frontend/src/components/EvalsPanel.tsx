import {
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
import type { EvalReport } from '../api/foreshock';

function pct(v?: number): string {
  return v == null ? '-' : `${(v * 100).toFixed(0)}%`;
}

function Metric({ label, value }: { label: string; value: string }) {
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
  report: EvalReport | null;
  running: boolean;
  onRun: () => void;
}

export default function EvalsPanel({ report, running, onRun }: Props) {
  const has = report != null && report.total != null;
  return (
    <Paper sx={{ p: 2 }}>
      <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 1 }}>
        <Box>
          <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
            Eval Harness
          </Typography>
          <Typography variant="caption" color="text.secondary">
            Fault scenario - expected diagnosis, retrieval precision/recall, hallucination check
          </Typography>
        </Box>
        <Button size="small" variant="contained" onClick={onRun} disabled={running}>
          {running ? 'Running...' : 'Run evals'}
        </Button>
      </Stack>

      {!has && (
        <Typography color="text.secondary">
          No eval runs yet. Click "Run evals" to score the LLM/RAG layer.
        </Typography>
      )}

      {has && report && (
        <Box>
          <Stack direction="row" flexWrap="wrap" sx={{ mb: 1 }}>
            <Metric label="diagnosis accuracy" value={pct(report.diagnosis_accuracy)} />
            <Metric label="retrieval precision" value={pct(report.retrieval_precision)} />
            <Metric label="retrieval recall" value={pct(report.retrieval_recall)} />
            <Metric label="hallucination" value={pct(report.hallucination_rate)} />
            <Metric label="passed" value={`${report.passed}/${report.total}`} />
          </Stack>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>case</TableCell>
                <TableCell>expected</TableCell>
                <TableCell>predicted</TableCell>
                <TableCell align="right">P</TableCell>
                <TableCell align="right">R</TableCell>
                <TableCell align="center">halluc.</TableCell>
                <TableCell align="center">result</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {(report.details ?? []).map((d, i) => (
                <TableRow key={i}>
                  <TableCell>{d.sample_id}</TableCell>
                  <TableCell>{d.expected}</TableCell>
                  <TableCell>{d.predicted}</TableCell>
                  <TableCell align="right">{d.retrieval_precision}</TableCell>
                  <TableCell align="right">{d.retrieval_recall}</TableCell>
                  <TableCell align="center">{d.hallucinated ? 'yes' : 'no'}</TableCell>
                  <TableCell align="center">
                    <Chip
                      size="small"
                      color={d.passed ? 'success' : 'error'}
                      label={d.passed ? 'pass' : 'fail'}
                    />
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
