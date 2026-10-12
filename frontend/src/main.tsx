import React from "react";
import ReactDOM from "react-dom/client";
import "./styles.css";

function AppPlaceholder() {
  return (
    <main className="app-shell">
      <section className="app-placeholder">
        <p className="eyebrow">RAG WORKSPACE</p>
        <h1>Knowledge workspace</h1>
        <p>The chat workspace is loading.</p>
      </section>
    </main>
  );
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <AppPlaceholder />
  </React.StrictMode>,
);

