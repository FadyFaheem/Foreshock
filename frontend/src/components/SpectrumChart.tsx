import { useMemo } from 'react';
import { Box, Paper, Typography } from '@mui/material';
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

interface Props {
  title: string;
  f: number[];
  mag: number[];
  color: string;
  faultFrequencies?: Record<string, number>;
}

export default function SpectrumChart({
  title,
  f,
  mag,
  color,
  faultFrequencies,
}: Props) {
  const data = useMemo(
    () => f.map((freq, i) => ({ f: freq, mag: mag[i] })),
    [f, mag],
  );

  return (
    <Paper sx={{ p: 2, height: '100%' }}>
      <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>
        {title}
      </Typography>
      <Box sx={{ width: '100%', height: 260 }}>
        <ResponsiveContainer>
          {/* top margin leaves room for the fault-frequency labels (BPFO, ...) */}
          <LineChart data={data} margin={{ top: 26, right: 16, bottom: 18, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey="f"
              type="number"
              domain={['dataMin', 'dataMax']}
              tickFormatter={(v) => `${Math.round(Number(v))}`}
              label={{ value: 'Frequency (Hz)', position: 'insideBottom', offset: -8 }}
              tick={{ fontSize: 12 }}
            />
            <YAxis tick={{ fontSize: 12 }} width={52} />
            <Tooltip
              formatter={(v) => Number(v).toExponential(2)}
              labelFormatter={(v) => `${Number(v).toFixed(1)} Hz`}
            />
            <Line
              type="monotone"
              dataKey="mag"
              stroke={color}
              dot={false}
              isAnimationActive={false}
              strokeWidth={1}
            />
            {faultFrequencies &&
              Object.entries(faultFrequencies).map(([name, freq]) => (
                <ReferenceLine
                  key={name}
                  x={freq}
                  stroke="#d32f2f"
                  strokeDasharray="4 4"
                  label={{
                    value: name,
                    fontSize: 10,
                    position: 'top',
                    fill: '#d32f2f',
                  }}
                />
              ))}
          </LineChart>
        </ResponsiveContainer>
      </Box>
    </Paper>
  );
}
