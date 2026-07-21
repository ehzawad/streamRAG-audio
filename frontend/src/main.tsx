import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { App } from "./App";
import { Home, NotFound } from "./Home";
import { modeForPath, normalizedPath } from "./routes.ts";
import "./styles.css";

const pathname = normalizedPath(window.location.pathname);
const mode = modeForPath(pathname);
const page = pathname === "/"
  ? <Home />
  : mode
    ? <App mode={mode} />
    : <NotFound />;

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    {page}
  </StrictMode>,
);
