import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

vi.mock("../state/store", () => ({
  useStore: () => ({
    session: {
      seed: 20260101,
      progress: { percent: 19 },
      dataset: {
        name: "Airline passengers",
        n_observations: 144,
        frequency: { label: "Monthly (start of month)" },
      },
    },
    sessionId: "session-1",
    health: { optional_features: {} },
    aiLiveReady: false,
    startSession: vi.fn(),
    endSession: vi.fn(),
    refresh: vi.fn(),
    openAssistant: vi.fn(),
    history: { can_undo: false, can_redo: false },
    historyBusy: false,
    undoHistory: vi.fn(),
    redoHistory: vi.fn(),
    refreshHistory: vi.fn(),
  }),
}));

import { TopBar } from "./TopBar";

describe("TopBar overflow menu", () => {
  it("keeps secondary actions reachable without a clipped right edge", () => {
    const html = renderToStaticMarkup(<TopBar />);
    expect(html).toContain('data-testid="topbar-more"');
    expect(html).toContain("topbar-more-menu");
    expect(html).toContain('data-testid="open-assistant"');
    expect(html).toContain("More actions");
    expect(html).toContain('data-testid="open-help"');
    expect(html).toContain('data-testid="open-error-library"');
    expect(html).toContain('data-testid="history-undo"');
    expect(html).toContain('data-testid="topbar-more-glossary"');
    expect(html).toContain('data-testid="topbar-more-simplify"');
    expect(html).toContain('data-testid="topbar-more-plain"');
    expect(html).toContain("pill-session");
  });
});
