import { useMemo } from 'react';
import { Box, Paper, Typography } from '@mui/material';
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { BRAND_COLORS } from '../theme/theme';

interface Props {
  t: number[];
  x: number[];
}

export default function WaveformChart({ t, x }: Props) {
  const data = useMemo(
    () => t.map((time, i) => ({ t: time, x: x[i] })),
    [t, x],
  );

  return (
    <Paper sx={{ p: 2 }}>
      <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>
        Raw waveform
      </Typography>
      <Box sx={{ width: '100%', height: 220 }}>
        <ResponsiveContainer>
          <LineChart data={data} margin={{ top: 5, right: 16, bottom: 18, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey="t"
              type="number"
              domain={['dataMin', 'dataMax']}
              tickFormatter={(v) => Number(v).toFixed(2)}
              label={{ value: 'Time (s)', position: 'insideBottom', offset: -8 }}
              tick={{ fontSize: 12 }}
            />
            <YAxis tick={{ fontSize: 12 }} width={52} />
            <Tooltip
              formatter={(v) => Number(v).toFixed(4)}
              labelFormatter={(v) => `t = ${Number(v).toFixed(4)} s`}
            />
            <Line
              type="monotone"
              dataKey="x"
              stroke={BRAND_COLORS.primary}
              dot={false}
              isAnimationActive={false}
              strokeWidth={1}
            />
          </LineChart>
        </ResponsiveContainer>
      </Box>
    </Paper>
  );
}
