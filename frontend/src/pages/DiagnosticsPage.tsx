import { useCallback, useEffect, useState } from 'react';
import {
  Alert,
  Box,
  Button,
  Chip,
  FormControl,
  InputLabel,
  MenuItem,
  Paper,
  Select,
  Stack,
} from '@mui/material';
import type { SelectChangeEvent } from '@mui/material';
import AgentWorkflow from '../components/AgentWorkflow';
import DiagnosisCard from '../components/DiagnosisCard';
import EvalsPanel from '../components/EvalsPanel';
import ObservabilityPanel from '../components/ObservabilityPanel';
import {
  diagnose,
  getAIStatus,
  getEvals,
  getObservability,
  getSamples,
  runAgent,
  runEvals,
  type AgentRun,
  type AIStatus,
  type Diagnosis,
  type EvalReport,
  type Observability,
  type Sample,
} from '../api/foreshock';

function msg(e: unknown): string {
  return e instanceof Error ? e.message : 'Request failed';
}

export default function DiagnosticsPage() {
  const [samples, setSamples] = useState<Sample[]>([]);
  const [selectedId, setSelectedId] = useState('');
  const [status, setStatus] = useState<AIStatus | null>(null);
  const [diag, setDiag] = useState<Diagnosis | null>(null);
  const [agentRun, setAgentRun] = useState<AgentRun | null>(null);
  const [obs, setObs] = useState<Observability | null>(null);
  const [evalReport, setEvalReport] = useState<EvalReport | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refreshObs = useCallback(() => {
    getObservability().then(setObs).catch(() => undefined);
  }, []);

  useEffect(() => {
    getSamples()
      .then((s) => {
        setSamples(s);
        if (s.length > 0) {
          // default to a fault sample so the agent has something to act on
          setSelectedId(s.find((x) => x.id !== 'normal')?.id ?? s[0].id);
        }
      })
      .catch((e) => setError(msg(e)));
    getAIStatus().then(setStatus).catch(() => undefined);
    getEvals().then(setEvalReport).catch(() => undefined);
    refreshObs();
  }, [refreshObs]);

  const onDiagnose = async () => {
    setBusy('diagnose');
    setError(null);
    try {
      setDiag(await diagnose(selectedId));
      refreshObs();
    } catch (e) {
      setError(msg(e));
    } finally {
      setBusy(null);
    }
  };

  const onAgent = async () => {
    setBusy('agent');
    setError(null);
    try {
      const r = await runAgent(selectedId);
      setAgentRun(r);
      setDiag(r.diagnosis);
      refreshObs();
    } catch (e) {
      setError(msg(e));
    } finally {
      setBusy(null);
    }
  };

  const onRunEvals = async () => {
    setBusy('evals');
    setError(null);
    try {
      setEvalReport(await runEvals());
      refreshObs();
    } catch (e) {
      setError(msg(e));
    } finally {
      setBusy(null);
    }
  };

  return (
    <Stack spacing={2}>
      <Paper sx={{ p: 2 }}>
        <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2} alignItems={{ sm: 'center' }}>
          <FormControl size="small" sx={{ minWidth: 240 }}>
            <InputLabel id="diag-sample">Sample signal</InputLabel>
            <Select
              labelId="diag-sample"
              label="Sample signal"
              value={selectedId}
              onChange={(e: SelectChangeEvent) => setSelectedId(e.target.value)}
            >
              {samples.map((s) => (
                <MenuItem key={s.id} value={s.id}>
                  {s.label}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          <Button variant="contained" onClick={onDiagnose} disabled={!selectedId || busy !== null}>
            {busy === 'diagnose' ? 'Diagnosing...' : 'Diagnose (RAG + LLM)'}
          </Button>
          <Button variant="outlined" onClick={onAgent} disabled={!selectedId || busy !== null}>
            {busy === 'agent' ? 'Running agent...' : 'Run agent'}
          </Button>
          {status && (
            <Stack direction="row" spacing={1} sx={{ ml: { sm: 'auto' } }}>
              <Chip
                size="small"
                variant="outlined"
                color={status.llm ? 'success' : 'default'}
                label={status.llm ? `LLM: ${status.model}` : 'LLM offline'}
              />
              <Chip
                size="small"
                variant="outlined"
                color={status.db ? 'success' : 'default'}
                label={`KB: ${status.kb_size}`}
              />
            </Stack>
          )}
        </Stack>
        {status && !status.llm && (
          <Alert severity="warning" sx={{ mt: 1.5 }}>
            LLM offline - diagnoses fall back to templated text. Start Ollama and pull the model.
          </Alert>
        )}
        {status && !status.db && (
          <Alert severity="warning" sx={{ mt: 1.5 }}>
            Database offline - retrieval, trend, and history are unavailable.
          </Alert>
        )}
      </Paper>

      {error && <Alert severity="error">{error}</Alert>}

      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' }, gap: 2 }}>
        <DiagnosisCard diagnosis={diag} loading={busy === 'diagnose' || busy === 'agent'} />
        <AgentWorkflow run={agentRun} loading={busy === 'agent'} />
      </Box>

      <EvalsPanel report={evalReport} running={busy === 'evals'} onRun={onRunEvals} />
      <ObservabilityPanel data={obs} onRefresh={refreshObs} />
    </Stack>
  );
}
