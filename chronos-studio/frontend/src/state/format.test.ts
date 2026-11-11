import { describe, expect, it } from "vitest";

import { pNumber, pPhrase } from "./format";

describe("p-value prose", () => {
  it("never claims an exact zero and never puts '=' before '<'", () => {
    expect(pPhrase(0)).toBe("p < 0.0001");
    expect(pPhrase(3.4e-120)).toBe("p < 0.0001");
    expect(pNumber(0)).toBe("< 0.0001");
  });

  it("keeps small but measurable values readable", () => {
    expect(pPhrase(0.0003703)).toBe("p = 0.00037");
    expect(pNumber(0.0003703)).toBe("0.00037");
    expect(pPhrase(0.0075)).toBe("p = 0.0075");
    expect(pPhrase(0.0076, 3)).toBe("p = 0.008");
    expect(pNumber(0.0076, 3)).toBe("0.008");
  });

  it("says when a value is missing rather than showing a fake number", () => {
    expect(pPhrase(null)).toContain("not available");
    expect(pNumber(undefined)).toBe("—");
    expect(pPhrase(Number.NaN)).toContain("not available");
  });
});
