import { describe, expect, it } from "vitest";

import { ApiError } from "../state/api";
import { GLOSSARY } from "./glossary";
import {
  ERROR_GROUPS,
  ERROR_LIBRARY,
  ERROR_PREFIX_HELP,
  FALLBACK_ERROR_HELP,
  errorHelp,
  errorSearch,
  errorToastPayload,
} from "./errors";

describe("Part 19 plain-English error library", () => {
  it("is a substantial, well-formed library with real glossary links", () => {
    expect(ERROR_LIBRARY.length).toBeGreaterThanOrEqual(30);
    expect(ERROR_PREFIX_HELP.length).toBeGreaterThanOrEqual(8);
    const codes = new Set<string>();
    for (const e of [...ERROR_LIBRARY, ...ERROR_PREFIX_HELP, FALLBACK_ERROR_HELP]) {
      expect(e.code.length).toBeGreaterThan(0);
      expect(ERROR_GROUPS).toContain(e.group);
      expect(e.title.length).toBeGreaterThan(15);
      expect(e.meaning.length).toBeGreaterThan(30);
      expect(e.fix.length).toBeGreaterThan(20);
      expect(e.avoid.length).toBeGreaterThan(20);
      expect(e.terms.length).toBeGreaterThanOrEqual(3);
      for (const slug of e.terms) expect(GLOSSARY[slug], `${e.code}: missing ${slug}`).toBeDefined();
      if (e !== FALLBACK_ERROR_HELP && !e.code.endsWith(".")) codes.add(e.code);
    }
    expect(codes.size).toBe(ERROR_LIBRARY.length);
  });

  it("resolves exact codes, prefix families and unknown codes", () => {
    expect(errorHelp("data.empty").title).toContain("no data rows");
    expect(errorHelp("session.stage_order").fix).toContain("stepper");
    expect(errorHelp("forecast.hierarchy_total").group).toBe("Models & forecasts");
    expect(errorHelp("catalog.chromium_missing").group).toBe("Evaluation & reports");
    expect(errorHelp("ai.key_missing").group).toBe("Assistant & privacy");
    expect(errorHelp("totally_new.code").code).toBe("totally_new.code");
    expect(errorHelp("totally_new.code").title).toBe(FALLBACK_ERROR_HELP.title);
    expect(errorHelp("").code).toBe("unknown");
  });

  it("searches code, prose and group", () => {
    const all = [...ERROR_LIBRARY, ...ERROR_PREFIX_HELP];
    expect(errorSearch(all, "").length).toBe(all.length);
    expect(errorSearch(all, "MAPE").map((e) => e.code)).toContain("evaluate.protocol_refused");
    expect(errorSearch(all, "data.empty").map((e) => e.code)).toContain("data.empty");
    expect(errorSearch(all, "", "Assistant & privacy").every((e) => e.group === "Assistant & privacy")).toBe(true);
    expect(errorSearch(all, "no_such_error_xyz")).toHaveLength(0);
  });

  it("turns an ApiError into a linked, plain-English toast", () => {
    const e = new ApiError(422, "data.non_numeric", "The column you chose holds text, not numbers.", "Pick a numeric column.");
    const t = errorToastPayload(e, "Fallback");
    expect(t.kind).toBe("error");
    expect(t.title).toContain("text, not numbers");
    expect(t.fix).toContain("numeric column");
    expect(t.errorCode).toBe("data.non_numeric");
    expect(t.body).toContain(errorHelp("data.non_numeric").meaning);
    const plain = errorToastPayload(new Error("local failure"), "That step failed.");
    expect(plain.title).toBe("That step failed.");
    expect(plain.errorCode).toBe("unknown");
  });
});
