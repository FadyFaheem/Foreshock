import {
  Box,
  Chip,
  LinearProgress,
  Paper,
  Stack,
  Step,
  StepLabel,
  Stepper,
  Typography,
} from '@mui/material';
import type { AgentRun } from '../api/foreshock';

const STEP_TITLES: Record<string, string> = {
  pull_signal: 'Pull signal',
  analyze: 'Analyze (features + classifier)',
  health_check: 'Health monitor (anomaly detector)',
  anomaly_check: 'Anomaly check',
  retrieve_kb: 'Retrieve knowledge (RAG)',
  check_trend: 'Check health trend',
  generate_diagnosis: 'Generate diagnosis',
  emit_work_order: 'Emit work order',
};

type ChipColor = 'default' | 'warning' | 'error';

function priorityColor(p: string): ChipColor {
  return p === 'high' ? 'error' : p === 'medium' ? 'warning' : 'default';
}

interface Props {
  run: AgentRun | null;
  loading?: boolean;
}

export default function AgentWorkflow({ run, loading }: Props) {
  return (
    <Paper sx={{ p: 2, height: '100%' }}>
      <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>
        Agentic Workflow
      </Typography>

      {loading && <LinearProgress sx={{ my: 2 }} />}
      {!loading && !run && (
        <Typography color="text.secondary">
          Run the agent to execute the multi-step diagnostic chain.
        </Typography>
      )}

      {run && (
        <Box>
          <Stepper orientation="vertical" activeStep={run.steps.length}>
            {run.steps.map((s, i) => (
              <Step key={i} completed>
                <StepLabel
                  optional={
                    <Typography variant="caption" color="text.secondary">
                      {s.detail}
                    </Typography>
                  }
                >
                  {STEP_TITLES[s.step] ?? s.step}
                  {s.status === 'skipped' && (
                    <Chip size="small" label="skipped" sx={{ ml: 1 }} />
                  )}
                </StepLabel>
              </Step>
            ))}
          </Stepper>

          {run.work_order && (
            <Box
              sx={{
                mt: 1,
                p: 1.5,
                bgcolor: 'grey.50',
                borderRadius: 1,
                border: '1px solid',
                borderColor: 'divider',
              }}
            >
              <Stack direction="row" spacing={1} alignItems="center">
                <Typography variant="body2" sx={{ fontWeight: 700 }}>
                  Draft work order #{run.work_order.id}
                </Typography>
                <Chip
                  size="small"
                  color={priorityColor(run.work_order.priority)}
                  label={`priority: ${run.work_order.priority}`}
                />
              </Stack>
              <Typography variant="caption" color="text.secondary">
                {run.work_order.asset} - {run.work_order.condition}
              </Typography>
              <Box component="ul" sx={{ m: '6px 0', pl: 2.5 }}>
                {run.work_order.actions.map((a, i) => (
                  <li key={i}>
                    <Typography variant="body2">{a}</Typography>
                  </li>
                ))}
              </Box>
            </Box>
          )}
        </Box>
      )}
    </Paper>
  );
}
