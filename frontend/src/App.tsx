import { useEffect, useRef, useState } from "react";

import {
  cancelTurn,
  commit,
  getServiceTopology,
  type BackendEvent,
  type PathName,
  sendSnapshot,
  type ServiceTopology,
  type Source,
  subscribe,
} from "./api";
import {
  isRunTransportTerminal,
  isUserVisibleRunComplete,
  recordUserVisibleTerminal,
  selectedImplementations,
  startSelectedRuns,
  type AnswerPath,
} from "./runLifecycle";
import { EXPERIENCE_PATHS } from "./routes.ts";

type PanelState = {
  answer: string;
  sources: Source[];
  status: string;
  firstToken: number | null;
  total: number | null;
  retrievalLead: number | null;
  candidateRetrievalLead: number | null;
  reuseMode: string | null;
  cacheHit: boolean | null;
  controllerCalls: number | null;
  retrievalCalls: number | null;
  toolCalls: number | null;
  fallbacks: number | null;
  persistenceStatus: string | null;
};

type SnapshotJob = {
  text: string;
  turnId: string;
  sessionId: string;
  revision: number;
  epoch: number;
  signal: AbortSignal;
};

type TranscriptAnswer = {
  answer: string;
  sources: Source[];
  status: string;
  firstToken: number | null;
  total: number | null;
};

type TranscriptTurn = {
  id: string;
  question: string;
  answers: Partial<Record<AnswerPath, TranscriptAnswer>>;
};

const emptyPanel = (status: string): PanelState => ({
  answer: "",
  sources: [],
  status,
  firstToken: null,
  total: null,
  retrievalLead: null,
  candidateRetrievalLead: null,
  reuseMode: null,
  cacheHit: null,
  controllerCalls: null,
  retrievalCalls: null,
  toolCalls: null,
  fallbacks: null,
  persistenceStatus: null,
});

const initialPanels = (mode: PathName): Record<AnswerPath, PanelState> => ({
  naive: emptyPanel(mode === "stream" ? "Not selected" : "Retrieval begins after Send."),
  stream: emptyPanel(mode === "naive" ? "Not selected" : "Waiting for a precise partial query."),
});

const freshIds = (): Record<AnswerPath, string> => ({
  naive: crypto.randomUUID(),
  stream: crypto.randomUUID(),
});

const freshAbortControllers = (): Record<AnswerPath, AbortController> => ({
  naive: new AbortController(),
  stream: new AbortController(),
});

const pendingTranscriptAnswer = (): TranscriptAnswer => ({
  answer: "",
  sources: [],
  status: "Waiting…",
  firstToken: null,
  total: null,
});

function isProbeReady(topology: ServiceTopology | null, path: AnswerPath): boolean {
  const probe = topology?.services[path];
  return Boolean(
    probe?.health.ok &&
      probe.health.index_ready &&
      probe.health.dataset_checksums_valid &&
      probe.health.index_matches_current_corpus &&
      probe.data.index_matches_current_corpus,
  );
}

function isModeReady(topology: ServiceTopology | null, mode: PathName): boolean {
  if (!topology) return false;
  if (mode === "compare" && topology.comparisonError) return false;
  return selectedImplementations(mode).every((path) => isProbeReady(topology, path));
}

function topologyLabel(topology: ServiceTopology | null, mode: PathName): string {
  if (!topology) return "Connecting to isolated services…";
  const selected = selectedImplementations(mode);
  const serviceError = selected
    .map((path) => topology.errors[path])
    .find((message): message is string => Boolean(message));
  if (serviceError) return serviceError;
  if (mode === "compare" && topology.comparisonError) return topology.comparisonError;
  const unavailable = selected.find((path) => !isProbeReady(topology, path));
  if (unavailable) {
    const probe = topology.services[unavailable];
    return `${unavailable} · ${probe?.health.dataset_status ?? "unavailable index"}`;
  }
  const stream = topology.services.stream?.health;
  const active = topology.services[selected[0]]?.health;
  if (!active) return "Service unavailable";
  return mode === "compare"
    ? `2 isolated APIs · ${active.model} · ${active.indexed_chunks} matched points each · Stream prefetch ${stream?.settled_draft_delay_ms ?? "—"} ms`
    : `${active.implementation} API · ${active.model} · ${active.indexed_chunks} chunks`;
}

export function App({ mode }: { mode: PathName }) {
  const [query, setQuery] = useState("");
  const [topology, setTopology] = useState<ServiceTopology | null>(null);
  const [transportNotice, setTransportNotice] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [trace, setTrace] = useState<string[]>([]);
  const [panels, setPanels] = useState<Record<AnswerPath, PanelState>>(() => initialPanels(mode));
  const [conversation, setConversation] = useState<TranscriptTurn[]>([]);

  const sessionIds = useRef(freshIds());
  const turnIds = useRef(freshIds());
  const turnOpened = useRef<Record<AnswerPath, boolean>>({ naive: false, stream: false });
  const revisions = useRef<Record<AnswerPath, number>>({ naive: 0, stream: 0 });
  const userVisibleTerminalPaths = useRef<Set<AnswerPath>>(new Set());
  const activeMode = useRef<PathName | null>(null);
  const visibleRunFinalized = useRef(true);
  const submissionEpoch = useRef(0);
  const editingAfterRun = useRef(false);

  const timer = useRef<number | undefined>(undefined);
  const pendingQuery = useRef("");
  const lastSnapshotText = useRef("");
  const lastQueuedText = useRef("");
  const queuedSnapshot = useRef<SnapshotJob | null>(null);
  const snapshotDraining = useRef(false);
  const snapshotAbort = useRef(new AbortController());
  const snapshotEpoch = useRef(0);
  const turnEvents = useRef<EventSource | null>(null);

  const runAbort = useRef(freshAbortControllers());
  const runEvents = useRef<Record<AnswerPath, EventSource | null>>({ naive: null, stream: null });
  const displayedRunIds = useRef<Record<AnswerPath, string | null>>({ naive: null, stream: null });

  useEffect(() => {
    snapshotAbort.current = new AbortController();
    runAbort.current = freshAbortControllers();
    queuedSnapshot.current = null;
    snapshotDraining.current = false;
    turnEvents.current = null;
    runEvents.current = { naive: null, stream: null };
    const controller = new AbortController();
    let retry: number | undefined;
    setTopology(null);

    async function refreshTopology() {
      const selected = selectedImplementations(mode);
      const value = await getServiceTopology(selected, controller.signal);
      if (controller.signal.aborted) return;
      setTopology(value);
      if (!isModeReady(value, mode)) {
        retry = window.setTimeout(() => void refreshTopology(), 2_000);
      }
    }

    void refreshTopology();
    return () => {
      controller.abort();
      window.clearTimeout(retry);
      window.clearInterval(timer.current);
      snapshotAbort.current.abort();
      Object.values(runAbort.current).forEach((controller) => controller.abort());
      queuedSnapshot.current = null;
      turnEvents.current?.close();
      Object.values(runEvents.current).forEach((source) => source?.close());
      (["naive", "stream"] as AnswerPath[]).forEach((path) => {
        if (turnOpened.current[path]) {
          void cancelTurn(path, turnIds.current[path]).catch(() => undefined);
        }
      });
    };
  }, [mode]);

  useEffect(() => {
    if (mode === "naive" || running) {
      timer.current = undefined;
      return;
    }
    const interval = window.setInterval(() => {
      const latest = pendingQuery.current.trim();
      if (!latest) {
        queuedSnapshot.current = null;
        lastQueuedText.current = "";
        return;
      }
      if (latest === lastQueuedText.current) return;
      lastQueuedText.current = latest;
      queueSnapshot(latest);
    }, 400);
    timer.current = interval;
    return () => {
      window.clearInterval(interval);
      if (timer.current === interval) timer.current = undefined;
    };
  }, [mode, running]);

  const ready = isModeReady(topology, mode);
  const health = transportNotice ?? topologyLabel(topology, mode);

  function addTrace(event: BackendEvent) {
    setTrace((current) => [
      ...current.slice(-39),
      `${event.type}${event.path ? ` · ${event.path}` : ""}`,
    ]);
  }

  function handleTypedEvent(event: BackendEvent) {
    addTrace(event);
    if (event.type === "trigger.decision") {
      setPanels((current) => ({
        ...current,
        stream: { ...current.stream, status: `${event.action}${event.query ? `: ${event.query}` : ""}` },
      }));
    } else if (event.type === "draft.settled") {
      setPanels((current) => ({
        ...current,
        stream: {
          ...current.stream,
          status: event.state === "ready"
            ? "Exact draft evidence ready — press Send for the grounded answer."
            : "Draft settled; retrieving exact text before Send…",
        },
      }));
    } else if (event.type === "retrieval.started") {
      setPanels((current) => ({
        ...current,
        stream: {
          ...current.stream,
          status: event.candidate
            ? "Retrieving a completed-prefix candidate…"
            : "Retrieving before submit…",
        },
      }));
    } else if (event.type === "retrieval.ready") {
      setPanels((current) => ({
        ...current,
        stream: {
          ...current.stream,
          status: event.commit_safe_exact
            ? "Exact draft evidence ready — press Send for the grounded answer."
            : event.candidate
              ? "Candidate evidence ready; validating intent…"
              : "Evidence ready before submit.",
        },
      }));
    } else if (event.type === "retrieval.discarded") {
      setPanels((current) => ({
        ...current,
        stream: { ...current.stream, status: "Stale evidence discarded." },
      }));
    } else if (event.type === "retrieval.revalidated") {
      setPanels((current) => ({
        ...current,
        stream: {
          ...current.stream,
          status: "Evidence validated and ready — press Send for the grounded answer.",
        },
      }));
    } else if (event.type === "retrieval.reused") {
      setPanels((current) => ({
        ...current,
        stream: {
          ...current.stream,
          status: event.ready_before_commit
            ? "Using evidence ready before Send."
            : event.retrieval_completed_before_commit
              ? "Using retrieval ready before Send and revalidated at Send."
              : "Using speculative retrieval that finished after Send.",
        },
      }));
    } else if (event.type === "retrieval.fallback") {
      setPanels((current) => ({
        ...current,
        stream: { ...current.stream, status: "Retrieving safely at Send…" },
      }));
    }
  }

  function finishVisibleRunIfReady() {
    const submittedMode = activeMode.current;
    if (
      !submittedMode ||
      visibleRunFinalized.current ||
      !isUserVisibleRunComplete(submittedMode, userVisibleTerminalPaths.current)
    ) return;

    visibleRunFinalized.current = true;
    editingAfterRun.current = true;
    setRunning(false);
    window.clearInterval(timer.current);
    timer.current = undefined;
    turnEvents.current?.close();
    turnEvents.current = null;
    snapshotAbort.current.abort();
    snapshotAbort.current = new AbortController();
    queuedSnapshot.current = null;
    snapshotEpoch.current += 1;
    lastSnapshotText.current = "";
    lastQueuedText.current = pendingQuery.current.trim();

    selectedImplementations(submittedMode).forEach((path) => {
      turnOpened.current[path] = false;
      turnIds.current[path] = crypto.randomUUID();
      revisions.current[path] = 0;
    });
  }

  function updateTranscript(
    transcriptTurnId: string,
    path: AnswerPath,
    update: (answer: TranscriptAnswer) => TranscriptAnswer,
  ) {
    setConversation((current) => current.map((turn) => {
      if (turn.id !== transcriptTurnId) return turn;
      const answer = turn.answers[path] ?? pendingTranscriptAnswer();
      return {
        ...turn,
        answers: { ...turn.answers, [path]: update(answer) },
      };
    }));
  }

  function handleRunEvent(
    path: AnswerPath,
    event: BackendEvent,
    transcriptTurnId: string,
  ) {
    addTrace(event);
    const wasUserVisibleTerminal = userVisibleTerminalPaths.current.has(path);
    if (event.type === "answer.started") {
      updateTranscript(transcriptTurnId, path, (answer) => ({
        ...answer,
        sources: event.sources || answer.sources,
        status: "Generating…",
      }));
    } else if (event.type === "answer.delta") {
      updateTranscript(transcriptTurnId, path, (answer) => ({
        ...answer,
        answer: answer.answer + (event.text || ""),
        status: "Generating…",
      }));
    } else if (event.type === "answer.ready" || event.type === "answer.completed") {
      updateTranscript(transcriptTurnId, path, (answer) => ({
        ...answer,
        answer: event.answer || answer.answer,
        sources: event.sources || answer.sources,
        status: "Complete",
        firstToken: event.timing?.submit_to_first_token_ms ?? answer.firstToken,
        total: event.timing?.total_response_ms ?? answer.total,
      }));
    } else if (event.type === "answer.error" || event.type === "run.error") {
      updateTranscript(transcriptTurnId, path, (answer) => ({
        ...answer,
        status: event.message || `${path} path failed`,
      }));
    }
    setPanels((current) => {
      const panel = current[path];
      if (event.type === "answer.started") {
        return {
          ...current,
          [path]: { ...emptyPanel("Generating grounded answer…"), sources: event.sources || [] },
        };
      }
      if (event.type === "answer.delta") {
        return { ...current, [path]: { ...panel, answer: panel.answer + (event.text || "") } };
      }
      if (event.type === "answer.ready" || event.type === "answer.completed") {
        return {
          ...current,
          [path]: {
            ...panel,
            answer: event.answer || panel.answer,
            sources: event.sources || panel.sources,
            status: "Complete",
            firstToken: event.timing?.submit_to_first_token_ms ?? panel.firstToken,
            total: event.timing?.total_response_ms ?? panel.total,
            retrievalLead:
              event.timing?.accepted_retrieval_lead_at_commit_ms ?? panel.retrievalLead,
            candidateRetrievalLead:
              event.timing?.accepted_candidate_retrieval_lead_ms ?? panel.candidateRetrievalLead,
            reuseMode: event.reuse?.mode ?? panel.reuseMode,
            cacheHit: event.retrieval?.cache_hit ?? panel.cacheHit,
            controllerCalls: event.controller?.calls ?? panel.controllerCalls,
            retrievalCalls: event.retrieval?.calls ?? panel.retrievalCalls,
            toolCalls: event.tool_traces?.length ?? panel.toolCalls,
            fallbacks: event.reuse?.commit_fallbacks ?? panel.fallbacks,
            persistenceStatus: event.type === "answer.ready"
              ? "pending"
              : event.persistence?.status ?? panel.persistenceStatus,
          },
        };
      }
      if (event.type === "answer.error" || event.type === "run.error") {
        return {
          ...current,
          [path]: wasUserVisibleTerminal
            ? { ...panel, persistenceStatus: "failed" }
            : { ...panel, status: event.message || `${path} path failed` },
        };
      }
      return current;
    });

    recordUserVisibleTerminal(userVisibleTerminalPaths.current, {
      type: event.type,
      path,
    });
    finishVisibleRunIfReady();
  }

  async function snapshot(job: SnapshotJob) {
    const normalized = job.text.trim();
    if (
      job.epoch !== snapshotEpoch.current ||
      job.turnId !== turnIds.current.stream ||
      !normalized ||
      normalized === lastSnapshotText.current
    ) return;

    turnOpened.current.stream = true;
    const accepted = await sendSnapshot({
      turnId: job.turnId,
      sessionId: job.sessionId,
      revision: job.revision,
      text: normalized,
      signal: job.signal,
    });
    setTransportNotice(null);
    if (!turnEvents.current) {
      turnEvents.current = subscribe(
        "stream",
        accepted.events_url,
        (event) => {
          if (
            job.epoch === snapshotEpoch.current &&
            job.turnId === turnIds.current.stream
          ) handleTypedEvent(event);
        },
        () => setTransportNotice("Typed-input event stream interrupted; reconnecting…"),
      );
    }
    if (
      job.epoch === snapshotEpoch.current &&
      job.turnId === turnIds.current.stream
    ) lastSnapshotText.current = normalized;
  }

  async function drainSnapshots() {
    if (snapshotDraining.current) return;
    snapshotDraining.current = true;
    try {
      while (queuedSnapshot.current) {
        const job = queuedSnapshot.current;
        queuedSnapshot.current = null;
        try {
          await snapshot(job);
        } catch (error) {
          if (error instanceof Error && error.name !== "AbortError") {
            setTransportNotice(error.message);
          }
        }
      }
    } finally {
      snapshotDraining.current = false;
      if (queuedSnapshot.current) void drainSnapshots();
    }
  }

  function queueSnapshot(text: string) {
    revisions.current.stream += 1;
    queuedSnapshot.current = {
      text,
      turnId: turnIds.current.stream,
      sessionId: sessionIds.current.stream,
      revision: revisions.current.stream,
      epoch: snapshotEpoch.current,
      signal: snapshotAbort.current.signal,
    };
    void drainSnapshots();
  }

  function onQuery(value: string) {
    setQuery(value);
    pendingQuery.current = value;
    const normalized = value.trim();
    const queued = queuedSnapshot.current;
    if (queued && queued.text.trim() !== normalized) {
      queuedSnapshot.current = null;
      lastQueuedText.current = normalized === lastSnapshotText.current
        ? normalized
        : lastSnapshotText.current;
    }
    if (editingAfterRun.current) {
      editingAfterRun.current = false;
      selectedImplementations(mode).forEach((path) => {
        displayedRunIds.current[path] = null;
      });
      setTrace([]);
      setPanels(initialPanels(mode));
    }
  }

  function attachRunEvents(
    path: AnswerPath,
    runId: string,
    eventsUrl: string,
    transcriptTurnId: string,
  ) {
    displayedRunIds.current[path] = runId;
    runEvents.current[path]?.close();
    let transportTerminal = false;
    const source = subscribe(
      path,
      eventsUrl,
      (event) => {
        if (displayedRunIds.current[path] === runId) {
          handleRunEvent(path, event, transcriptTurnId);
        }
        if (isRunTransportTerminal(event)) {
          transportTerminal = true;
          source.close();
          if (runEvents.current[path] === source) runEvents.current[path] = null;
        }
      },
      () => {
        if (transportTerminal) return;
        if (displayedRunIds.current[path] !== runId) source.close();
        else setTransportNotice(`${path} answer event stream interrupted; reconnecting…`);
      },
    );
    runEvents.current[path] = source;
  }

  async function submit() {
    const text = query.trim();
    if (!text || running || !ready) return;

    const submittedMode = mode;
    const epoch = submissionEpoch.current + 1;
    submissionEpoch.current = epoch;
    activeMode.current = submittedMode;
    visibleRunFinalized.current = false;
    editingAfterRun.current = false;
    setTransportNotice(null);
    userVisibleTerminalPaths.current.clear();
    setRunning(true);
    setTrace([]);
    setPanels({
      naive: emptyPanel(submittedMode === "stream" ? "Not selected" : "Working…"),
      stream: emptyPanel(submittedMode === "naive" ? "Not selected" : "Working…"),
    });

    window.clearInterval(timer.current);
    timer.current = undefined;
    snapshotAbort.current.abort();
    snapshotAbort.current = new AbortController();
    queuedSnapshot.current = null;
    snapshotEpoch.current += 1;
    turnEvents.current?.close();
    turnEvents.current = null;

    const selected = selectedImplementations(submittedMode);
    const transcriptTurnId = crypto.randomUUID();
    setConversation((current) => [
      ...current,
      {
        id: transcriptTurnId,
        question: text,
        answers: Object.fromEntries(
          selected.map((path) => [path, pendingTranscriptAnswer()]),
        ) as Partial<Record<AnswerPath, TranscriptAnswer>>,
      },
    ]);
    setQuery("");
    pendingQuery.current = "";
    const queryTime = new Date().toISOString();
    const requests = Object.fromEntries(selected.map((path) => {
      runAbort.current[path].abort();
      runAbort.current[path] = new AbortController();
      revisions.current[path] += 1;
      turnOpened.current[path] = true;
      return [path, {
        turnId: turnIds.current[path],
        sessionId: sessionIds.current[path],
        revision: revisions.current[path],
        signal: runAbort.current[path].signal,
      }];
    })) as Partial<Record<AnswerPath, {
      turnId: string;
      sessionId: string;
      revision: number;
      signal: AbortSignal;
    }>>;

    const starts = await startSelectedRuns(submittedMode, async (path) => {
      const request = requests[path];
      if (!request) throw new Error(`missing ${path} request lifecycle`);
      const accepted = await commit({
        implementation: path,
        ...request,
        text,
        queryTime,
      });
      if (submissionEpoch.current !== epoch) {
        void cancelTurn(path, request.turnId).catch(() => undefined);
        throw new DOMException("superseded request", "AbortError");
      }
      attachRunEvents(path, accepted.run_id, accepted.events_url, transcriptTurnId);
      return accepted;
    });

    if (submissionEpoch.current !== epoch) return;
    starts.forEach((result) => {
      if (result.status === "rejected") {
        const message = result.reason instanceof Error
          ? result.reason.message
          : String(result.reason);
        handleRunEvent(
          result.path,
          { type: "run.error", path: result.path, message },
          transcriptTurnId,
        );
      }
    });
  }

  function newChat() {
    submissionEpoch.current += 1;
    activeMode.current = null;
    visibleRunFinalized.current = true;
    editingAfterRun.current = false;
    turnEvents.current?.close();
    turnEvents.current = null;
    snapshotAbort.current.abort();
    snapshotAbort.current = new AbortController();
    queuedSnapshot.current = null;
    snapshotEpoch.current += 1;

    (["naive", "stream"] as AnswerPath[]).forEach((path) => {
      const abandonedTurn = turnIds.current[path];
      if (turnOpened.current[path]) {
        void cancelTurn(path, abandonedTurn).catch(() => undefined);
      }
      runAbort.current[path].abort();
      runAbort.current[path] = new AbortController();
      runEvents.current[path]?.close();
      runEvents.current[path] = null;
      displayedRunIds.current[path] = null;
      turnOpened.current[path] = false;
    });

    turnIds.current = freshIds();
    sessionIds.current = freshIds();
    revisions.current = { naive: 0, stream: 0 };
    userVisibleTerminalPaths.current.clear();
    pendingQuery.current = "";
    lastSnapshotText.current = "";
    lastQueuedText.current = "";
    setRunning(false);
    setTransportNotice(null);
    setQuery("");
    setTrace([]);
    setPanels(initialPanels(mode));
    setConversation([]);
  }

  return (
    <main>
      <nav className="app-nav" aria-label="Application navigation">
        <a href="/">Home</a>
        {(Object.entries(EXPERIENCE_PATHS) as [PathName, string][]).map(([path, href]) => (
          <a
            key={path}
            href={href}
            aria-current={mode === path ? "page" : undefined}
            onClick={(event) => {
              if (mode === path) event.preventDefault();
            }}
          >
            {path === "naive" ? "Naive" : path === "stream" ? "Stream" : "Compare"}
          </a>
        ))}
      </nav>
      <header>
        <div>
          <p className="eyebrow">StreamRAG</p>
          <h1>Naive RAG vs StreamRAG</h1>
          <p className="subhead">
            Two isolated APIs use the same checksum-bound corpus and common answer contract.
            StreamRAG can retrieve while you type; Naive RAG starts at Send. Compare fans the same
            committed question out to both services concurrently, and neither service may show an
            answer before Send.
          </p>
        </div>
        <span className={`health ${ready ? "ok" : "warn"}`}>{health}</span>
      </header>

      <ConversationTranscript mode={mode} turns={conversation} />

      <section className="composer">
        <textarea
          aria-label="Question"
          value={query}
          onChange={(event) => onQuery(event.target.value)}
          placeholder="Type a factual question…"
          disabled={running}
        />
        <div className="actions">
          <div className="modes" aria-label="Retrieval path">
            {(Object.entries(EXPERIENCE_PATHS) as [PathName, string][]).map(([path, href]) => (
              <a
                key={path}
                href={href}
                className={mode === path ? "active" : ""}
                aria-current={mode === path ? "page" : undefined}
                onClick={(event) => {
                  if (mode === path) event.preventDefault();
                }}
              >
                {path === "naive" ? "Path A" : path === "stream" ? "Path B" : "Compare"}
              </a>
            ))}
          </div>
          <div className="submit-row">
            <button type="button" className="secondary" onClick={newChat}>
              New chat
            </button>
            <button
              type="button"
              className="primary"
              onClick={submit}
              disabled={!ready || !query.trim() || running}
            >
              {running ? "Running…" : "Send"}
            </button>
          </div>
        </div>
      </section>

      {mode === "compare" && <CompareSummary naive={panels.naive} stream={panels.stream} />}

      <section className={`results ${mode === "compare" ? "" : "single"}`}>
        {mode !== "stream" && (
          <ResultPanel path="naive" title="Path A · Naive RAG" tone="amber" panel={panels.naive} />
        )}
        {mode !== "naive" && (
          <ResultPanel path="stream" title="Path B · StreamRAG" tone="green" panel={panels.stream} />
        )}
      </section>

      <details>
        <summary>Event trace</summary>
        <pre>{trace.length ? trace.join("\n") : "No events yet."}</pre>
      </details>
    </main>
  );
}

function ConversationTranscript({
  mode,
  turns,
}: {
  mode: PathName;
  turns: TranscriptTurn[];
}) {
  if (!turns.length) return null;
  const paths = selectedImplementations(mode);
  return (
    <section className="conversation" aria-label="Conversation" aria-live="polite">
      <div className="conversation-head">
        <h2>Conversation</h2>
        <span>Follow-ups keep this chat&apos;s context.</span>
      </div>
      {turns.map((turn) => (
        <article className="chat-turn" key={turn.id}>
          <div className="user-message">
            <small>You</small>
            <p>{turn.question}</p>
          </div>
          <div className={`assistant-messages ${paths.length === 1 ? "single" : ""}`}>
            {paths.map((path) => {
              const answer = turn.answers[path] ?? pendingTranscriptAnswer();
              const visibleSources = Array.from(
                new Map(
                  answer.sources.map((source) => [source.url || source.chunk_id, source]),
                ).values(),
              );
              return (
                <div className={`assistant-message ${path}`} key={path}>
                  <div>
                    <small>{path === "naive" ? "Naive RAG" : "StreamRAG"}</small>
                    <span>{answer.status}</span>
                  </div>
                  <p className={answer.answer ? "" : "pending-answer"}>
                    {answer.answer || "Waiting for the grounded answer…"}
                  </p>
                  {(answer.firstToken !== null || answer.total !== null) && (
                    <p className="chat-timing">
                      TTFT {answer.firstToken?.toFixed(0) ?? "—"} ms · total {answer.total?.toFixed(0) ?? "—"} ms
                    </p>
                  )}
                  <div className="sources">
                    {visibleSources.map((source) => (
                      <a key={source.chunk_id} href={source.url} target="_blank" rel="noreferrer">
                        {source.title}
                      </a>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
        </article>
      ))}
    </section>
  );
}

function CompareSummary({ naive, stream }: { naive: PanelState; stream: PanelState }) {
  const complete = naive.status === "Complete" && stream.status === "Complete";
  const naiveCalls = totalCalls(naive);
  const streamCalls = totalCalls(stream);
  return (
    <aside className="compare-summary" aria-label="Live comparison scope">
      <div>
        <strong>Live diagnostic, not an accuracy score.</strong>
        <span>
          The browser starts two independent service runs at the same Send boundary. Inspect both
          answers and evidence; reportable latency and correctness come from the external benchmark.
        </span>
      </div>
      {complete && (
        <div className="deltas" aria-label="StreamRAG minus Naive RAG">
          <Delta label="TTFT" value={difference(stream.firstToken, naive.firstToken, "ms")} />
          <Delta label="Total" value={difference(stream.total, naive.total, "ms")} />
          <Delta label="Evidence docs" value={signed(uniqueSourceCount(stream.sources) - uniqueSourceCount(naive.sources))} />
          <Delta label="Calls" value={difference(streamCalls, naiveCalls)} />
        </div>
      )}
    </aside>
  );
}

function ResultPanel({
  path,
  title,
  tone,
  panel,
}: {
  path: AnswerPath;
  title: string;
  tone: string;
  panel: PanelState;
}) {
  const visibleSources = Array.from(
    new Map(panel.sources.map((source) => [source.url || source.chunk_id, source])).values(),
  );
  return (
    <article className={`panel ${tone}`}>
      <div className="panel-head">
        <h2>{title}</h2>
        <span>{panel.status}</span>
      </div>
      <div className="metrics">
        <Metric label="Path TTFT" value={panel.firstToken === null ? "—" : `${panel.firstToken.toFixed(0)} ms`} />
        <Metric label="Total" value={panel.total === null ? "—" : `${panel.total.toFixed(0)} ms`} />
        <Metric label="Accepted lead" value={panel.retrievalLead == null ? "—" : `${panel.retrievalLead.toFixed(0)} ms`} />
        <Metric label="Retrieval-ready lead" value={panel.candidateRetrievalLead == null ? "—" : `${panel.candidateRetrievalLead.toFixed(0)} ms`} />
        <Metric label="Evidence" value={evidenceLabel(path, panel)} />
        <Metric label="Path cache" value={panel.cacheHit == null ? "—" : panel.cacheHit ? "Hit" : "Miss"} />
        <Metric label="Evidence docs" value={panel.status === "Complete" ? String(visibleSources.length) : "—"} />
        <Metric label="Persistence" value={persistenceLabel(panel.persistenceStatus)} />
        <Metric label="Calls · ctl / ret / tool" value={callBreakdown(panel)} />
        <Metric label="Send fallbacks" value={panel.fallbacks == null ? "—" : String(panel.fallbacks)} />
      </div>
      <div className={`answer ${panel.answer ? "" : "empty"}`}>{panel.answer || "No answer yet."}</div>
      <div className="sources">
        {visibleSources.map((source) => (
          <a key={source.chunk_id} href={source.url} target="_blank" rel="noreferrer">
            {source.title}
          </a>
        ))}
      </div>
    </article>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return <div><small>{label}</small><strong>{value}</strong></div>;
}

function Delta({ label, value }: { label: string; value: string }) {
  return <span><small>{label}</small><strong>{value}</strong></span>;
}

function uniqueSourceCount(sources: Source[]) {
  return new Set(sources.map((source) => source.url || source.chunk_id)).size;
}

function persistenceLabel(status: string | null) {
  if (status === "pending") return "Finalizing…";
  if (status === "completed") return "Saved";
  if (status === "timeout") return "Timed out";
  if (status === "failed") return "Failed";
  return "—";
}

function totalCalls(panel: PanelState) {
  const calls = [panel.controllerCalls, panel.retrievalCalls, panel.toolCalls];
  return calls.every((value) => value === null)
    ? null
    : calls.reduce<number>((total, value) => total + (value ?? 0), 0);
}

function callBreakdown(panel: PanelState) {
  const calls = [panel.controllerCalls, panel.retrievalCalls, panel.toolCalls];
  return calls.every((value) => value === null)
    ? "—"
    : calls.map((value) => value ?? 0).join(" / ");
}

function evidenceLabel(path: AnswerPath, panel: PanelState) {
  if (panel.reuseMode === "precommit_exact") return "Ready before Send";
  if (panel.reuseMode === "precommit_revalidated") return "Revalidated";
  if (panel.reuseMode === "presubmit_retrieval_revalidated_at_commit") return "Prefetched · checked at Send";
  if (panel.reuseMode === "inflight_completed_postcommit") return "In-flight overlap";
  if (panel.reuseMode === "committed_text_retrieval") {
    return path === "stream" && panel.fallbacks ? "Send fallback" : "At Send";
  }
  return "—";
}

function difference(left: number | null, right: number | null, unit = "") {
  if (left === null || right === null) return "—";
  const value = left - right;
  return `${signed(Math.round(value))}${unit ? ` ${unit}` : ""}`;
}

function signed(value: number) {
  if (value > 0) return `+${value}`;
  if (value < 0) return `−${Math.abs(value)}`;
  return "0";
}
