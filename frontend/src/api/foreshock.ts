import { apiJson } from './client';

export interface Sample {
  id: string;
  condition: string;
  label: string;
}

export interface Series {
  f: number[];
  mag: number[];
}

export interface SignalResponse {
  id: string;
  condition: string;
  label: string;
  fs: number;
  rpm: number;
  fault_frequencies: Record<string, number>;
  waveform: { t: number[]; x: number[] };
  spectrum: Series;
  envelope: Series;
}

export interface ClassProbability {
  condition: string;
  label: string;
  probability: number;
}

export interface FeatureValue {
  name: string;
  value: number;
}

export interface PredictionResponse {
  prediction: string;
  prediction_label: string;
  confidence: number;
  n_windows: number;
  rpm: number;
  probabilities: ClassProbability[];
  features: FeatureValue[];
}

export function getSamples(): Promise<Sample[]> {
  return apiJson<Sample[]>('/api/samples');
}

export function getSignal(id: string): Promise<SignalResponse> {
  return apiJson<SignalResponse>(`/api/signal/${encodeURIComponent(id)}`);
}

export function predictSample(sampleId: string): Promise<PredictionResponse> {
  const form = new FormData();
  form.append('sample_id', sampleId);
  return apiJson<PredictionResponse>('/api/predict', {
    method: 'POST',
    body: form,
  });
}

export function predictFile(file: File): Promise<PredictionResponse> {
  const form = new FormData();
  form.append('file', file);
  return apiJson<PredictionResponse>('/api/predict', {
    method: 'POST',
    body: form,
  });
}

export interface RandomTestResult {
  actual: string;
  actual_label: string;
  prediction: string;
  prediction_label: string;
  correct: boolean;
  confidence: number;
  probabilities: ClassProbability[];
  rpm: number;
  noise: number;
  waveform: { t: number[]; x: number[] };
}

export function randomTest(condition: string, noise: number): Promise<RandomTestResult> {
  const form = new FormData();
  form.append('condition', condition);
  form.append('noise', String(noise));
  return apiJson<RandomTestResult>('/api/random_test', { method: 'POST', body: form });
}

export interface InjectBase {
  signal: number[];
  fs: number;
  rpm: number;
}

export interface HealthVerdict {
  error: number;
  threshold: number;
  caught: boolean;
}

export interface InjectResult {
  prediction: string;
  prediction_label: string;
  confidence: number;
  classifier_caught: boolean;
  caught: boolean;
  n_points: number;
  amplitude: number;
  probabilities: ClassProbability[];
  health: HealthVerdict | null;
  waveform: { t: number[]; x: number[] };
}

export function getInjectBase(): Promise<InjectBase> {
  return apiJson<InjectBase>('/api/inject/base');
}

export function injectFaults(
  signal: number[],
  points: number[],
  amplitude: number,
  fs: number,
  rpm: number,
): Promise<InjectResult> {
  return apiJson<InjectResult>('/api/inject', {
    method: 'POST',
    body: JSON.stringify({ signal, points, amplitude, fs, rpm }),
  });
}

// --- AI layer: RAG diagnosis, agent, evals, observability -----------------

export interface AIStatus {
  db: boolean;
  llm: boolean;
  kb_size: number;
  model: string;
  embed_model: string;
}

export interface Source {
  title: string;
  source: string;
  fault_type: string;
  score: number;
}

export interface Diagnosis {
  id?: number;
  sample_id?: string;
  asset: string;
  condition: string;
  label: string;
  confidence: number;
  rpm: number;
  rms?: number;
  severity: string;
  summary: string;
  likely_cause?: string;
  recommended_actions: string[];
  priority: string;
  sources: Source[];
  used_llm: boolean;
  probabilities: Record<string, number>;
}

export interface AgentStep {
  step: string;
  status: string;
  detail: string;
}

export interface TrendPoint {
  rms: number;
  condition: string;
  created_at: string;
}

export interface Trend {
  direction: string;
  summary: string;
  delta_pct?: number;
  history: TrendPoint[];
}

export interface WorkOrder {
  id?: number;
  asset: string;
  condition: string;
  priority: string;
  actions: string[];
  status: string;
}

export interface AgentRun {
  asset: string;
  anomaly: boolean;
  steps: AgentStep[];
  trend: Trend;
  diagnosis: Diagnosis;
  work_order: WorkOrder | null;
}

export interface ObsSummary {
  calls: number;
  avg_latency_ms: number;
  total_tokens: number;
  avg_retrieval_score: number;
  errors: number;
}

export interface ObsByOp {
  operation: string;
  calls: number;
  avg_latency_ms: number;
  total_tokens: number;
}

export interface ObsCall {
  operation: string;
  model: string;
  latency_ms: number;
  prompt_tokens: number | null;
  completion_tokens: number | null;
  total_tokens: number | null;
  retrieval_score: number | null;
  ok: boolean;
  created_at: string;
}

export interface Observability {
  summary: ObsSummary;
  by_operation: ObsByOp[];
  recent: ObsCall[];
}

export interface EvalDetail {
  sample_id: string;
  expected: string;
  predicted: string;
  confidence: number;
  correct: boolean;
  retrieval_precision: number;
  retrieval_recall: number;
  hallucinated: boolean;
  used_llm: boolean;
  passed: boolean;
  top_source: string | null;
}

export interface EvalReport {
  id?: number;
  suite?: string;
  total?: number;
  passed?: number;
  diagnosis_accuracy?: number;
  retrieval_precision?: number;
  retrieval_recall?: number;
  hallucination_rate?: number;
  details?: EvalDetail[];
  created_at?: string;
}

export function getAIStatus(): Promise<AIStatus> {
  return apiJson<AIStatus>('/api/ai/status');
}

export function diagnose(sampleId: string): Promise<Diagnosis> {
  const form = new FormData();
  form.append('sample_id', sampleId);
  return apiJson<Diagnosis>('/api/diagnose', { method: 'POST', body: form });
}

export function runAgent(sampleId: string): Promise<AgentRun> {
  const form = new FormData();
  form.append('sample_id', sampleId);
  return apiJson<AgentRun>('/api/agent', { method: 'POST', body: form });
}

export interface InjectDiagnoseResult {
  agent: AgentRun;
  requested_fault: string | null;
  injected_points: number[];
  amplitude: number;
  // Just the characteristic frequency of the detected fault, for the UI marker.
  marked_frequency: Record<string, number>;
  fault_frequencies: Record<string, number>;
  waveform: { t: number[]; x: number[] };
  envelope: Series;
}

export interface InjectDiagnoseRequest {
  signal: number[];
  fs: number;
  rpm: number;
  // Mode A: generate a realistic fault of this type (periodic impulse train).
  fault_type?: string;
  severity?: number;
  // Mode B: inject isolated bursts at these manually placed sample indices.
  points?: number[];
  amplitude?: number;
  asset?: string;
}

// Generate/inject a fault into a healthy window, then run the full RAG + LLM agent
// on the synthesized signal. Powers the Fault Lab page.
export function injectDiagnose(
  req: InjectDiagnoseRequest,
): Promise<InjectDiagnoseResult> {
  return apiJson<InjectDiagnoseResult>('/api/inject/diagnose', {
    method: 'POST',
    body: JSON.stringify(req),
  });
}

export function getObservability(): Promise<Observability> {
  return apiJson<Observability>('/api/observability');
}

export function getEvals(): Promise<EvalReport> {
  return apiJson<EvalReport>('/api/evals');
}

export function runEvals(): Promise<EvalReport> {
  return apiJson<EvalReport>('/api/evals/run', { method: 'POST' });
}

// --- v2: health indicator (autoencoder) -----------------------------------

export interface HealthTrendPoint {
  i: number;
  error: number;
  smooth: number;
  phase: string;
}

export interface HealthTrend {
  source: string;
  threshold: number;
  alarm_index: number;
  points: HealthTrendPoint[];
}

export interface EmbeddingPoint {
  x: number;
  y: number;
  condition: string;
  label: string;
}

export interface HealthEmbedding {
  points: EmbeddingPoint[];
}

export function getHealthTrend(): Promise<HealthTrend> {
  return apiJson<HealthTrend>('/api/health/trend');
}

export function getHealthEmbedding(): Promise<HealthEmbedding> {
  return apiJson<HealthEmbedding>('/api/health/embedding');
}

// --- v3: live sensor feed (Kafka) -----------------------------------------

export interface StreamStatus {
  kafka: boolean;
  topic: string;
  recent: number;
  streaming: boolean;
}

export interface StreamEvent {
  ts: number;
  actual: string | null;
  actual_label: string | null;
  prediction: string;
  prediction_label: string;
  confidence: number;
  rms: number;
  correct: boolean | null;
}

export function getStreamStatus(): Promise<StreamStatus> {
  return apiJson<StreamStatus>('/api/stream/status');
}

export function simulateStream(count: number, interval: number): Promise<{ status: string }> {
  return apiJson('/api/stream/simulate', {
    method: 'POST',
    body: JSON.stringify({ count, interval }),
  });
}

export function stopStream(): Promise<{ status: string }> {
  return apiJson('/api/stream/stop', { method: 'POST' });
}
