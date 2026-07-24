import {
  COMMIT_REQUEST_TIMEOUT_MS,
  SNAPSHOT_REQUEST_TIMEOUT_MS,
  SSE_EVENT_TYPES,
  type AnswerPath,
} from "./runLifecycle.ts";

export type PathName = AnswerPath | "compare";

export type BackendEvent = {
  type: string;
  run_id?: string;
  path?: AnswerPath;
  text?: string;
  answer?: string;
  action?: string;
  state?: string;
  query?: string | null;
  message?: string;
  sources?: Source[];
  timing?: {
    submit_to_first_token_ms: number | null;
    total_response_ms: number;
    accepted_retrieval_lead_at_commit_ms: number;
    accepted_candidate_retrieval_lead_ms: number;
  };
  retrieval?: {
    cache_hit: boolean;
    calls: number;
  };
  controller?: {
    calls: number;
  };
  reuse?: {
    mode:
      | "precommit_exact"
      | "precommit_revalidated"
      | "presubmit_retrieval_revalidated_at_commit"
      | "inflight_completed_postcommit"
      | "committed_text_retrieval";
    commit_fallbacks: number;
  };
  tool_traces?: unknown[];
  ready_before_commit?: boolean;
  retrieval_completed_before_commit?: boolean;
  candidate?: boolean;
  commit_safe_exact?: boolean;
  persistence?: {
    status: string;
    elapsed_ms: number | null;
  };
};

export type Source = {
  chunk_id: string;
  title: string;
  url: string;
  score: number;
};

export type ServiceHealth = {
  ok: boolean;
  implementation: AnswerPath;
  metrics_contract_version: number;
  supports_snapshots: boolean;
  dataset_status: string;
  indexed_chunks: number;
  indexed_desired_chunks: number;
  index_ready: boolean;
  dataset_checksums_valid: boolean;
  index_matches_current_corpus: boolean;
  model: string;
  embedding_model: string;
  settled_draft_delay_ms?: number;
  instance_id: string;
};

export type ServiceDataStatus = {
  implementation: AnswerPath;
  metrics_contract_version: number;
  supports_snapshots: boolean;
  approval_status: string;
  indexed_chunks: number;
  indexed_desired_chunks: number;
  index_checksum: string;
  index_source_sha256: string;
  current_index_source_sha256: string;
  index_matches_current_corpus: boolean;
  model: string;
  embedding_model: string;
  settled_draft_delay_ms?: number;
  configuration: Record<string, unknown>;
  config_hash: string;
  serving_dataset_checksum: string;
  documents_sha256: string;
};

export type ServiceProbe = {
  health: ServiceHealth;
  data: ServiceDataStatus;
};

export type ServiceTopology = {
  services: Partial<Record<AnswerPath, ServiceProbe>>;
  errors: Partial<Record<AnswerPath, string>>;
  comparisonError: string | null;
};

const SERVICE_URLS: Record<AnswerPath, string> = {
  naive: "/api/naive",
  stream: "/api/stream",
};

export const METRICS_CONTRACT_VERSION = 2;
const TOPOLOGY_REQUEST_TIMEOUT_MS = 5_000;

const COMMON_IDENTITY_FIELDS = [
  "config_hash",
  "serving_dataset_checksum",
  "documents_sha256",
  "index_checksum",
  "index_source_sha256",
  "current_index_source_sha256",
  "indexed_chunks",
  "indexed_desired_chunks",
  "model",
  "embedding_model",
  "configuration",
] as const satisfies readonly (keyof ServiceDataStatus)[];

export function serviceBaseUrl(implementation: AnswerPath): string {
  return SERVICE_URLS[implementation];
}

async function fetchJson<T>(
  implementation: AnswerPath,
  path: string,
  signal?: AbortSignal,
): Promise<T> {
  const controller = new AbortController();
  const abort = () => controller.abort();
  if (signal?.aborted) abort();
  else signal?.addEventListener("abort", abort, { once: true });
  const timeout = globalThis.setTimeout(abort, TOPOLOGY_REQUEST_TIMEOUT_MS);
  try {
    const response = await fetch(`${serviceBaseUrl(implementation)}${path}`, {
      signal: controller.signal,
    });
    if (!response.ok) throw new Error(`${implementation} ${path} failed: ${response.status}`);
    return response.json() as Promise<T>;
  } catch (error) {
    if (controller.signal.aborted && !signal?.aborted) {
      throw new Error(`${implementation} ${path} timed out`);
    }
    throw error;
  } finally {
    globalThis.clearTimeout(timeout);
    signal?.removeEventListener("abort", abort);
  }
}

async function post<T>(
  implementation: AnswerPath,
  path: string,
  body: unknown,
  signal?: AbortSignal,
  timeoutMs = 10_000,
): Promise<T> {
  const controller = new AbortController();
  const abort = () => controller.abort();
  if (signal?.aborted) abort();
  else signal?.addEventListener("abort", abort, { once: true });
  const timeout = globalThis.setTimeout(abort, timeoutMs);
  try {
    const response = await fetch(`${serviceBaseUrl(implementation)}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    if (!response.ok) {
      const detail = await response.text();
      throw new Error(`${implementation} ${response.status}: ${detail}`);
    }
    return response.json() as Promise<T>;
  } finally {
    globalThis.clearTimeout(timeout);
    signal?.removeEventListener("abort", abort);
  }
}

function assertServiceContract(
  implementation: AnswerPath,
  health: ServiceHealth,
  data: ServiceDataStatus,
): void {
  if (health.implementation !== implementation || data.implementation !== implementation) {
    throw new Error(`${implementation} URL returned the wrong implementation role`);
  }
  if (
    health.metrics_contract_version !== METRICS_CONTRACT_VERSION ||
    data.metrics_contract_version !== METRICS_CONTRACT_VERSION
  ) {
    throw new Error(`${implementation} uses an incompatible metrics contract`);
  }
  const expectedSnapshots = implementation === "stream";
  if (
    health.supports_snapshots !== expectedSnapshots ||
    data.supports_snapshots !== expectedSnapshots
  ) {
    throw new Error(`${implementation} reports an invalid snapshot capability`);
  }
  const triggerFields = [health.settled_draft_delay_ms, data.settled_draft_delay_ms];
  if (implementation === "stream" && triggerFields.some((value) => value === undefined)) {
    throw new Error("stream does not advertise its typed-input configuration");
  }
  if (implementation === "naive" && triggerFields.some((value) => value !== undefined)) {
    throw new Error("naive advertises Stream-only configuration");
  }
}

export async function probeService(
  implementation: AnswerPath,
  signal?: AbortSignal,
): Promise<ServiceProbe> {
  const [health, data] = await Promise.all([
    fetchJson<ServiceHealth>(implementation, "/v1/health", signal),
    fetchJson<ServiceDataStatus>(implementation, "/v1/data/status", signal),
  ]);
  assertServiceContract(implementation, health, data);
  return { health, data };
}

function comparableValue(value: unknown): string {
  return typeof value === "object" ? JSON.stringify(value) : String(value);
}

export function validateCommonIdentity(
  naive: ServiceDataStatus,
  stream: ServiceDataStatus,
): void {
  for (const field of COMMON_IDENTITY_FIELDS) {
    const naiveValue = comparableValue(naive[field]);
    const streamValue = comparableValue(stream[field]);
    if (!naiveValue || naiveValue === "undefined" || naiveValue !== streamValue) {
      throw new Error(`service identity mismatch: ${field}`);
    }
  }
}

export function validateServiceIsolation(
  naive: ServiceProbe,
  stream: ServiceProbe,
  naiveUrl = serviceBaseUrl("naive"),
  streamUrl = serviceBaseUrl("stream"),
): void {
  const baseUrl = typeof window === "undefined" ? "http://localhost" : window.location.origin;
  const normalizedNaiveUrl = new URL(naiveUrl, baseUrl).href.replace(/\/$/, "");
  const normalizedStreamUrl = new URL(streamUrl, baseUrl).href.replace(/\/$/, "");
  if (normalizedNaiveUrl === normalizedStreamUrl) {
    throw new Error("Naive and Stream must use distinct service URLs");
  }
  const naiveInstance = naive.health.instance_id.trim();
  const streamInstance = stream.health.instance_id.trim();
  if (!naiveInstance || !streamInstance) {
    throw new Error("both services must advertise a non-empty instance ID");
  }
  if (naiveInstance === streamInstance) {
    throw new Error("Naive and Stream must use distinct service instances");
  }
}

export async function getServiceTopology(
  implementations: readonly AnswerPath[] = ["naive", "stream"],
  signal?: AbortSignal,
): Promise<ServiceTopology> {
  const settled = await Promise.allSettled(
    implementations.map((implementation) => probeService(implementation, signal)),
  );
  const services: Partial<Record<AnswerPath, ServiceProbe>> = {};
  const errors: Partial<Record<AnswerPath, string>> = {};

  settled.forEach((result, index) => {
    const implementation = implementations[index];
    if (result.status === "fulfilled") services[implementation] = result.value;
    else errors[implementation] =
      result.reason instanceof Error ? result.reason.message : String(result.reason);
  });

  let comparisonError: string | null = null;
  if (implementations.length === 2 && services.naive && services.stream) {
    try {
      validateServiceIsolation(services.naive, services.stream);
      validateCommonIdentity(services.naive.data, services.stream.data);
    } catch (error) {
      comparisonError = error instanceof Error ? error.message : String(error);
    }
  } else if (implementations.length === 2) {
    comparisonError = "both isolated services are required for comparison";
  }
  return { services, errors, comparisonError };
}

export function sendSnapshot(args: {
  turnId: string;
  sessionId: string;
  revision: number;
  text: string;
  signal?: AbortSignal;
}) {
  return post<{ turn_id: string; revision: number; events_url: string }>(
    "stream",
    `/v1/turns/${args.turnId}/snapshots`,
    {
      session_id: args.sessionId,
      revision: args.revision,
      text: args.text,
    },
    args.signal,
    SNAPSHOT_REQUEST_TIMEOUT_MS,
  );
}

export async function commit(args: {
  implementation: AnswerPath;
  turnId: string;
  sessionId: string;
  revision: number;
  text: string;
  queryTime: string;
  signal?: AbortSignal;
}) {
  const accepted = await post<{
    run_id: string;
    turn_id: string;
    path: AnswerPath;
    events_url: string;
  }>(
    args.implementation,
    `/v1/turns/${args.turnId}/commit`,
    {
      session_id: args.sessionId,
      revision: args.revision,
      text: args.text,
      query_time: args.queryTime,
    },
    args.signal,
    COMMIT_REQUEST_TIMEOUT_MS,
  );
  if (accepted.path !== args.implementation) {
    throw new Error(`${args.implementation} commit returned path ${accepted.path}`);
  }
  return accepted;
}

export async function cancelTurn(implementation: AnswerPath, turnId: string) {
  const response = await fetch(
    `${serviceBaseUrl(implementation)}/v1/turns/${turnId}`,
    { method: "DELETE", keepalive: true },
  );
  if (!response.ok && response.status !== 404) {
    throw new Error(`${implementation} turn cancellation failed: ${response.status}`);
  }
}

export function subscribe(
  implementation: AnswerPath,
  path: string,
  onEvent: (event: BackendEvent) => void,
  onTransportError?: () => void,
) {
  const source = new EventSource(`${serviceBaseUrl(implementation)}${path}`);
  const dispatch = (data: string) => {
    const event = JSON.parse(data) as BackendEvent;
    onEvent({ ...event, path: implementation });
  };
  source.onerror = () => onTransportError?.();
  source.onmessage = (event) => dispatch(event.data);
  SSE_EVENT_TYPES.forEach((name) =>
    source.addEventListener(name, (event) => dispatch((event as MessageEvent).data)),
  );
  return source;
}
