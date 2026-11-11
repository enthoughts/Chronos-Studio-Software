import { describe, expect, it } from "vitest";

import {
  catalogCell,
  catalogFormatLabel,
  catalogGroupCounts,
  catalogWithReport,
  filterCatalogItems,
} from "./CatalogPanel";
import type {
  CatalogItem,
  CatalogReportResponse,
  CatalogResponse,
  CatalogScorecard,
} from "../state/types";

function item(overrides: Partial<CatalogItem> = {}): CatalogItem {
  return {
    id: "fig.series",
    title: "Observed series",
    group: "Data & quality",
    stage: "S1",
    kind: "plot",
    source: "session.results[S1]",
    caption: "The observed series over time.",
    alt_text: "Line chart of 144 observations.",
    figure_json: { data: [], layout: {} },
    columns: [{ key: "time", label: "Time", dtype: "string", align: "left" }],
    rows: [{ time: "1949-01-01" }],
    formats: ["png", "svg", "pdf", "xlsx", "csv"],
    dpi: 300,
    report_insertable: true,
    ...overrides,
  } as CatalogItem;
}

const items: CatalogItem[] = [
  item(),
  item({ id: "table.observations", title: "Observation register", kind: "table", group: "Tables" }),
  item({ id: "models.var_irf", title: "VAR impulse responses", group: "Model fit", stage: "S10" }),
];

function scorecard(): CatalogScorecard {
  return {
    plots: 49,
    tables: 49,
    min_plots: 35,
    min_tables: 30,
    plots_ok: true,
    tables_ok: true,
    captions_ok: true,
    alt_text_ok: true,
    data_behind_ok: true,
    groups: {
      "Data & quality": { plots: 13, tables: 0 },
      "Tables": { plots: 0, tables: 49 },
    },
    styling: {
      single_source: "app.charts.theme",
      mode: "light",
      colorblind_safe: true,
      dark_light_print: ["light", "dark", "print"],
      dash_patterns: true,
    },
    exports: {
      formats: ["png", "svg", "pdf", "xlsx", "csv"],
      dpi: 300,
      captioned: true,
      report_insertable: true,
      data_behind: true,
    },
  } as CatalogScorecard;
}

function catalogData(): CatalogResponse {
  return {
    status: "ok",
    stage: "S15",
    mode: "light",
    scorecard: scorecard(),
    groups: [],
    plots: items,
    tables: [],
    report: { item_ids: [], items: [], count: 0 },
    exports: {
      formats: ["png", "svg", "pdf", "xlsx", "csv"],
      dpi: 300,
      scale: 3.125,
      endpoint: "/analysis/catalog",
    },
  } as CatalogResponse;
}

describe("Part 16 catalogue filters", () => {
  it("filters by group, kind and a case-insensitive search", () => {
    expect(filterCatalogItems(items, "all", "all", "").map((x) => x.id))
      .toEqual(["fig.series", "table.observations", "models.var_irf"]);
    expect(filterCatalogItems(items, "all", "table", "").map((x) => x.id))
      .toEqual(["table.observations"]);
    expect(filterCatalogItems(items, "Model fit", "plot", "impulse").map((x) => x.id))
      .toEqual(["models.var_irf"]);
    expect(filterCatalogItems(items, "Tables", "plot", "")).toHaveLength(0);
  });

  it("searches captions and ids as well as titles", () => {
    expect(filterCatalogItems(items, "all", "all", "OBSERVATION").map((x) => x.id))
      .toContain("table.observations");
    expect(filterCatalogItems(items, "all", "all", "fig.series")).toHaveLength(1);
  });
});

describe("Part 16 report basket state", () => {
  it("uses the authoritative API response for add and clear without losing catalogue items", () => {
    const before = catalogData();
    const added: CatalogReportResponse = {
      status: "ok",
      action: "add",
      item_ids: ["fig.series"],
      items: [items[0]],
      count: 1,
    };
    const afterAdd = catalogWithReport(before, added);
    expect(afterAdd.report.count).toBe(1);
    expect(afterAdd.report.items[0].title).toBe("Observed series");
    expect(afterAdd.plots).toBe(before.plots);
    expect(before.report.count).toBe(0);

    const cleared: CatalogReportResponse = {
      status: "ok",
      action: "clear",
      item_ids: [],
      items: [],
      count: 0,
    };
    expect(catalogWithReport(afterAdd, cleared).report.count).toBe(0);
  });
});

describe("Part 16 catalogue scorecard and export contract", () => {
  it("keeps all nine group counts in a stable array", () => {
    const counts = catalogGroupCounts(scorecard());
    expect(counts).toHaveLength(2);
    expect(counts[0]).toEqual({ name: "Data & quality", plots: 13, tables: 0 });
    expect(counts[1]).toEqual({ name: "Tables", plots: 0, tables: 49 });
  });

  it("labels all five export formats, including 300 dpi PNG and data downloads", () => {
    expect(catalogFormatLabel("png")).toContain("300 dpi");
    expect(catalogFormatLabel("svg")).toBe("SVG");
    expect(catalogFormatLabel("pdf")).toBe("PDF");
    expect(catalogFormatLabel("xlsx")).toContain("data");
    expect(catalogFormatLabel("csv")).toContain("data");
  });

  it("renders missing, non-finite, boolean and nested cells without throwing", () => {
    expect(catalogCell(null)).toBe("—");
    expect(catalogCell(Number.NaN)).toBe("—");
    expect(catalogCell(2.5)).toBe("2.5");
    expect(catalogCell(true)).toBe("yes");
    expect(catalogCell({ a: 1 })).toBe('{"a":1}');
  });
});
