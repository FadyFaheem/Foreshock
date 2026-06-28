import { useEffect, useState } from 'react';
import { Alert, Box, Paper, Stack, Typography } from '@mui/material';
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from 'recharts';
import {
  getHealthEmbedding,
  getHealthTrend,
  type EmbeddingPoint,
  type HealthEmbedding,
  type HealthTrend,
} from '../api/foreshock';
import { BRAND_COLORS, CONDITION_COLORS } from '../theme/theme';

export default function HealthPage() {
  const [trend, setTrend] = useState<HealthTrend | null>(null);
  const [embed, setEmbed] = useState<HealthEmbedding | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getHealthTrend()
      .then(setTrend)
      .catch((e) => setError(e instanceof Error ? e.message : 'Failed to load health trend'));
    getHealthEmbedding().then(setEmbed).catch(() => undefined);
  }, []);

  const trendData = trend?.points.map((p) => ({ i: p.i, error: p.error, smooth: p.smooth })) ?? [];

  const byCond: Record<string, EmbeddingPoint[]> = {};
  embed?.points.forEach((p) => {
    (byCond[p.condition] ??= []).push(p);
  });

  return (
    <Stack spacing={2}>
      <Paper sx={{ p: 2 }}>
        <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
          Health indicator - autoencoder reconstruction error
        </Typography>
        <Typography variant="caption" color="text.secondary">
          An autoencoder trained on healthy data only. Reconstruction error stays low while the
          bearing is healthy and rises as it degrades - an unsupervised early-warning signal.
          {trend ? ` Source: ${trend.source.toUpperCase()}.` : ''}
        </Typography>
      </Paper>

      {error && <Alert severity="error">{error}</Alert>}

      <Paper sx={{ p: 2 }}>
        <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 2 }}>
          Reconstruction-error trend (run to failure, log scale)
        </Typography>
        <Box sx={{ width: '100%', height: 360 }}>
          <ResponsiveContainer>
            <LineChart data={trendData} margin={{ top: 10, right: 28, bottom: 32, left: 8 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="i"
                type="number"
                domain={['dataMin', 'dataMax']}
                tick={{ fontSize: 12 }}
                height={48}
                label={{ value: 'time (window index)', position: 'insideBottom', offset: -4 }}
              />
              <YAxis
                scale="log"
                domain={['auto', 'auto']}
                allowDataOverflow
                tick={{ fontSize: 12 }}
                width={56}
                tickFormatter={(v) => {
                  const n = Number(v);
                  if (n >= 1000) return `${(n / 1000).toFixed(0)}k`;
                  if (n >= 1) return `${n.toFixed(0)}`;
                  return n.toFixed(2);
                }}
              />
              <Tooltip formatter={(v) => Number(v).toFixed(3)} />
              <Legend verticalAlign="top" height={30} />
              {trend && (
                <ReferenceLine
                  y={trend.threshold}
                  stroke="#d32f2f"
                  strokeDasharray="5 5"
                  label={{ value: 'alarm threshold', position: 'insideTopRight', fontSize: 11, fill: '#d32f2f' }}
                />
              )}
              {trend && trend.alarm_index >= 0 && (
                <ReferenceLine
                  x={trend.alarm_index}
                  stroke={BRAND_COLORS.secondaryDark}
                  strokeDasharray="3 3"
                  label={{ value: 'fault detected', position: 'top', fontSize: 11, fill: BRAND_COLORS.secondaryDark }}
                />
              )}
              <Line type="monotone" dataKey="error" name="error" stroke={BRAND_COLORS.mediumGray} dot={false} strokeWidth={1} isAnimationActive={false} />
              <Line type="monotone" dataKey="smooth" name="smoothed" stroke={BRAND_COLORS.primary} dot={false} strokeWidth={2} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </Box>
      </Paper>

      <Paper sx={{ p: 2 }}>
        <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 2 }}>
          2-D embedding - healthy clustering, faults drifting away
        </Typography>
        <Box sx={{ width: '100%', height: 400 }}>
          <ResponsiveContainer>
            <ScatterChart margin={{ top: 10, right: 28, bottom: 32, left: 8 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" dataKey="x" name="PC1" tick={{ fontSize: 12 }} height={44} label={{ value: 'PC1', position: 'insideBottom', offset: -4 }} />
              <YAxis type="number" dataKey="y" name="PC2" tick={{ fontSize: 12 }} width={56} />
              <ZAxis range={[36, 36]} />
              <Tooltip cursor={{ strokeDasharray: '3 3' }} />
              <Legend verticalAlign="top" height={30} />
              {Object.entries(byCond).map(([cond, pts]) => (
                <Scatter
                  key={cond}
                  name={pts[0]?.label ?? cond}
                  data={pts}
                  fill={CONDITION_COLORS[cond] ?? BRAND_COLORS.mediumGray}
                  fillOpacity={0.6}
                />
              ))}
            </ScatterChart>
          </ResponsiveContainer>
        </Box>
      </Paper>
    </Stack>
  );
}
