import { QueryClientProvider } from "@tanstack/react-query";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { queryClient } from "./queryClient";
import { savedDoc } from "./state/doc";
import { useUI } from "./state/ui";
import "./styles.css";

// First launch (nothing saved): show the preset gallery (U12).
if (!savedDoc()) useUI.getState().set({ galleryOpen: true });
else useUI.getState().set({ layoutSeq: 1 });

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </StrictMode>,
);
