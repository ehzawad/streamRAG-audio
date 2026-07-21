import type { PathName } from "./api.ts";

export const EXPERIENCE_PATHS: Record<PathName, string> = {
  naive: "/naive",
  stream: "/stream",
  compare: "/compare",
};

export function normalizedPath(pathname: string): string {
  const withoutTrailingSlash = pathname.replace(/\/+$/, "");
  return withoutTrailingSlash || "/";
}

export function modeForPath(pathname: string): PathName | null {
  const normalized = normalizedPath(pathname);
  return (Object.entries(EXPERIENCE_PATHS) as [PathName, string][])
    .find(([, path]) => path === normalized)?.[0] ?? null;
}

export function pathForMode(mode: PathName): string {
  return EXPERIENCE_PATHS[mode];
}
