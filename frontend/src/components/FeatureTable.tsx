import {
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Typography,
} from '@mui/material';
import type { FeatureValue } from '../api/foreshock';

interface Props {
  features: FeatureValue[];
}

function formatValue(v: number): string {
  if (v === 0) return '0';
  const abs = Math.abs(v);
  if (abs >= 1000 || abs < 1e-3) return v.toExponential(3);
  return v.toFixed(4);
}

export default function FeatureTable({ features }: Props) {
  return (
    <Paper sx={{ p: 2, height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>
        Extracted features
      </Typography>

      {features.length === 0 ? (
        <Typography color="text.secondary" sx={{ mt: 1 }}>
          Feature values appear here after a prediction.
        </Typography>
      ) : (
        <TableContainer sx={{ maxHeight: 360 }}>
          <Table size="small" stickyHeader>
            <TableHead>
              <TableRow>
                <TableCell>Feature</TableCell>
                <TableCell align="right">Value</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {features.map((f) => (
                <TableRow key={f.name} hover>
                  <TableCell sx={{ fontFamily: 'monospace' }}>{f.name}</TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                    {formatValue(f.value)}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      )}
    </Paper>
  );
}
