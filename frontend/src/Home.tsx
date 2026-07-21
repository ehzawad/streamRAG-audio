import { EXPERIENCE_PATHS } from "./routes.ts";

const experiences = [
  {
    path: EXPERIENCE_PATHS.naive,
    label: "Naive RAG",
    tag: "Path A",
    description: "Retrieval starts only after Send. Use this as the conventional baseline.",
  },
  {
    path: EXPERIENCE_PATHS.stream,
    label: "StreamRAG",
    tag: "Path B",
    description: "Evidence can be prepared while you type; the answer still waits for Send.",
  },
  {
    path: EXPERIENCE_PATHS.compare,
    label: "Compare",
    tag: "A/B",
    description: "Send the same committed question to both isolated services and inspect them together.",
  },
] as const;

export function Home() {
  return (
    <main className="home">
      <header className="home-header">
        <p className="eyebrow">StreamRAG</p>
        <h1>Choose an experience</h1>
        <p className="subhead">
          One URL exposes three views over the same checksum-bound knowledge base. The Naive and
          Stream services remain separate behind the frontend.
        </p>
      </header>

      <nav className="experience-grid" aria-label="RAG experiences">
        {experiences.map((experience) => (
          <a href={experience.path} key={experience.path}>
            <span>{experience.tag}</span>
            <h2>{experience.label}</h2>
            <p>{experience.description}</p>
            <strong>Open experience →</strong>
          </a>
        ))}
      </nav>

      <p className="home-note">
        The browser uses same-origin API routes. Internal service ports are not part of the user
        experience.
      </p>
    </main>
  );
}

export function NotFound() {
  return (
    <main className="not-found">
      <p className="eyebrow">404</p>
      <h1>That experience does not exist.</h1>
      <a href="/">Return to the homepage</a>
    </main>
  );
}
