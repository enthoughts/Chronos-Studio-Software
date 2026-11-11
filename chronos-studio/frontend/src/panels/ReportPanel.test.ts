import { describe, expect, it } from "vitest";

import {
  reportExportPath,
  reportScoreText,
  reportStepLabel,
  verdictTone,
} from "./ReportPanel";
import type { ReportNextStep, ReportStatusName } from "../state/types";

describe("Part 18 report panel helpers", () => {
  it("maps the three exact verdict states to distinct tones", () => {
    const states: ReportStatusName[] = ["ready", "nearly", "not_ready"];
    expect(states.map(verdictTone)).toEqual([
      "report-tone-ok", "report-tone-warn", "report-tone-bad",
    ]);
  });

  it("keeps the weighted score and numbered next step readable", () => {
    expect(reportScoreText(90, 100)).toBe("90/100");
    const step: ReportNextStep = {
      number: 2,
      item_id: "acf_pacf",
      text: "Open ACF / PACF. Visit S6 — ACF / PACF & Next-Step Advisor.",
      panel_id: "S6",
      panel_label: "ACF / PACF & Next-Step Advisor",
      result_ids: ["table.correlogram_acf"],
    };
    expect(reportStepLabel(step)).toContain("2. Open ACF / PACF");
    expect(reportStepLabel(step)).toContain("S6");
  });

  it("builds a relative export URL with scope, theme and optional attachment", () => {
    expect(reportExportPath("docx", "print", "all")).toBe(
      "/api/v1/export/docx?mode=print&scope=all&download=0",
    );
    expect(reportExportPath("html", "dark", "selected", true)).toBe(
      "/api/v1/export/html?mode=dark&scope=selected&download=1",
    );
  });
});
