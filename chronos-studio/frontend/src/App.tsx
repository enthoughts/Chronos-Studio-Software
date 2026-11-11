/**
 * App.tsx — the shell: top bar, ribbon, pipeline stepper, options pane, results.
 *
 * Part 19 adds the beginner modes that span the whole shell: Simplify view,
 * plain-language mode, glossary/demo dialogs, a per-panel "What is this?"
 * drawer, a plain-English error library, undo/redo and glossary tooltips on
 * arbitrary prose. Layout lives in CSS, state lives in the store, and step
 * logic lives in panels/.
 */

import { useEffect, useMemo, useState } from "react";
import { AssistantDrawer } from "./components/AssistantDrawer";
import { DemoPanel } from "./components/DemoPanel";
import { ErrorLibraryPanel } from "./components/ErrorLibraryPanel";
import { GlossaryPanel } from "./components/GlossaryPanel";
import { GlossarySelection } from "./components/GlossarySelection";
import { HelpDrawer } from "./components/HelpDrawer";
import { OptionsPane } from "./components/OptionsPane";
import { ResultsPane, type S15View } from "./components/ResultsPane";
import { Ribbon } from "./components/Ribbon";
import { Stepper } from "./components/Stepper";
import { Toasts } from "./components/Toasts";
import { TopBar } from "./components/TopBar";
import { useStore } from "./state/store";

export default function App() {
  const { health, openAssistant, setActiveStage, activeStage } = useStore();
  const [s15View, setS15View] = useState<S15View>("report");
  const [simplifyView, setSimplifyView] = useState(false);
  const [plainLanguage, setPlainLanguage] = useState(false);
  const [glossaryOpen, setGlossaryOpen] = useState(false);
  const [demoOpen, setDemoOpen] = useState(false);
  const [helpPanel, setHelpPanel] = useState<string | null>(null);
  const [errorLibrary, setErrorLibrary] = useState<{ open: boolean; code: string }>({
    open: false,
    code: "",
  });

  /* An error toast can open the library pre-filtered to its own code. */
  useEffect(() => {
    const open = (e: Event) => {
      const detail = (e as CustomEvent<{ code?: string }>).detail;
      setErrorLibrary({ open: true, code: detail?.code ?? "" });
    };
    window.addEventListener("chronos:open-error-library", open);
    return () => window.removeEventListener("chronos:open-error-library", open);
  }, []);

  const features = useMemo(() => health?.optional_features ?? {}, [health]);
  const optionalCount = Object.values(features).filter(Boolean).length;

  return (
    <div
      className="shell"
      data-simplify={simplifyView ? "true" : "false"}
      data-plain={plainLanguage ? "true" : "false"}
    >
      <a className="skip-link" href="#workspace">Skip to the analysis workspace</a>
      <TopBar
        simplifyView={simplifyView}
        plainLanguage={plainLanguage}
        onToggleSimplify={() => setSimplifyView((v) => !v)}
        onTogglePlainLanguage={() => setPlainLanguage((v) => !v)}
        onOpenGlossary={() => setGlossaryOpen(true)}
        onOpenDemo={() => setDemoOpen(true)}
        onOpenHelp={() => setHelpPanel(activeStage)}
        onOpenErrors={() => setErrorLibrary({ open: true, code: "" })}
      />
      <Ribbon
        simplifyView={simplifyView}
        onCommand={(id, stage) => {
          if (id === "report" || id === "catalog") {
            setS15View(id === "catalog" ? "catalog" : "report");
            setActiveStage("S15");
            return true;
          }
          if (id === "ask" || id === "explain") {
            openAssistant({
              action: id === "ask" ? "ask" : "explain",
              panelId: stage,
              question: id === "explain" ? "Explain this panel's computed results." : "",
            });
            return true;
          }
          return false;
        }}
      />
      <Stepper />
      <div className="workspace" id="workspace">
        <OptionsPane />
        <ResultsPane
          s15View={s15View}
          onSetS15View={setS15View}
          onOpenHelp={setHelpPanel}
        />
      </div>
      <span hidden aria-hidden>{optionalCount}</span>
      <AssistantDrawer />
      {glossaryOpen && <GlossaryPanel onClose={() => setGlossaryOpen(false)} />}
      {demoOpen && <DemoPanel onClose={() => setDemoOpen(false)} />}
      {helpPanel !== null && (
        <HelpDrawer panelId={helpPanel} onClose={() => setHelpPanel(null)} />
      )}
      {errorLibrary.open && (
        <ErrorLibraryPanel
          initialCode={errorLibrary.code}
          onClose={() => setErrorLibrary({ open: false, code: "" })}
        />
      )}
      <GlossarySelection />
      <Toasts />
    </div>
  );
}
