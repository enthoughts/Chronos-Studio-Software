import { describe, expect, it } from "vitest";

import {
  figureValueSummary,
  formatSummaryValue,
  progressValueSummary,
  tableValueSummary,
} from "./accessible";
import type { TableResult } from "./types";

function table(): Pick<TableResult, "caption" | "columns" | "rows"> {
  return {
    caption: "Descriptives for AirPassengers",
    columns: [
      { key: "stat", label: "Statistic", dtype: "string", align: "left" },
      { key: "value", label: "Value", dtype: "number", align: "right" },
    ],
    rows: [
      { stat: "Mean", value: 280.2986111111111 },
      { stat: "SD", value: 119.966 },
      { stat: "N", value: 144 },
    ],
  };
}

describe("Part 19 screen-reader value summaries", () => {
  it("includes the real numbers, not just a caption", () => {
    const s = tableValueSummary(table());
    expect(s).toContain("3 rows, 2 columns");
    expect(s).toContain("Statistic, Value");
    expect(s).toContain("280.299");
    expect(s).toContain("Statistic Mean; Value 280.299");
    expect(s).toContain("Statistic N; Value 144");
  });

  it("caps the spoken list and handles an empty table", () => {
    expect(tableValueSummary(table(), 2).split(";").length).toBeLessThanOrEqual(2);
    const empty = tableValueSummary({ caption: "Empty", columns: [], rows: [] });
    expect(empty).toContain("No values are available");
  });

  it("formats missing / non-finite / boolean cells in words", () => {
    expect(formatSummaryValue(null)).toBe("missing");
    expect(formatSummaryValue(undefined)).toBe("missing");
    expect(formatSummaryValue(NaN)).toBe("not available");
    expect(formatSummaryValue(true)).toBe("yes");
    expect(formatSummaryValue(false)).toBe("no");
    expect(formatSummaryValue(1.23456789)).toBe("1.23457");
  });

  it("describes a figure with the series count and its alt text", () => {
    const s = figureValueSummary("AirPassengers rises from 112 to 432 passengers.", 3);
    expect(s).toContain("3 series");
    expect(s).toContain("112 to 432");
    expect(figureValueSummary("", 0)).toContain("no descriptive text");
  });

  it("announces progress in words", () => {
    expect(progressValueSummary(3, 17, "Fit models")).toBe(
      "3 of 17 steps complete. Now: Fit models.",
    );
    expect(progressValueSummary(0, 17, "")).toContain("Idle");
  });
});
