export type AnswerPath = "naive" | "stream";
export type RunMode = AnswerPath | "compare";

export type RunLifecycleEvent = {
  type: string;
  path?: AnswerPath;
};

export const POST_ANSWER_LEASE_BUDGET_MS = 10_000;
export const SNAPSHOT_REQUEST_TIMEOUT_MS = 12_000;
export const COMMIT_REQUEST_TIMEOUT_MS = 15_000;

export const SSE_EVENT_TYPES = [
  "input.ack",
  "draft.settled",
  "trigger.decision",
  "trigger.error",
  "retrieval.started",
  "retrieval.ready",
  "retrieval.discarded",
  "retrieval.revalidated",
  "retrieval.reused",
  "retrieval.fallback",
  "retrieval.error",
  "answer.started",
  "answer.delta",
  "answer.ready",
  "answer.completed",
  "answer.error",
  "agent.tool_started",
  "agent.tool_completed",
  "agent.context_compressed",
  "run.started",
  "run.completed",
  "run.error",
] as const;

export function recordUserVisibleTerminal(
  completedPaths: Set<AnswerPath>,
  event: RunLifecycleEvent,
): boolean {
  if (
    !event.path ||
    (event.type !== "answer.ready" &&
      event.type !== "answer.error" &&
      event.type !== "run.error") ||
    completedPaths.has(event.path)
  ) return false;
  completedPaths.add(event.path);
  return true;
}

export function selectedImplementations(mode: RunMode): AnswerPath[] {
  return mode === "compare" ? ["naive", "stream"] : [mode];
}

export type RunStartResult<T> =
  | { path: AnswerPath; status: "fulfilled"; value: T }
  | { path: AnswerPath; status: "rejected"; reason: unknown };

export async function startSelectedRuns<T>(
  mode: RunMode,
  start: (path: AnswerPath) => Promise<T>,
): Promise<RunStartResult<T>[]> {
  return Promise.all(
    selectedImplementations(mode).map(async (path): Promise<RunStartResult<T>> => {
      try {
        return { path, status: "fulfilled", value: await start(path) };
      } catch (reason) {
        return { path, status: "rejected", reason };
      }
    }),
  );
}

export function isUserVisibleRunComplete(
  mode: RunMode,
  completedPaths: ReadonlySet<AnswerPath>,
): boolean {
  if (mode === "compare") {
    return completedPaths.has("naive") && completedPaths.has("stream");
  }
  return completedPaths.has(mode);
}

export function isRunTransportTerminal(event: RunLifecycleEvent): boolean {
  return event.type === "run.completed" || event.type === "run.error";
}
