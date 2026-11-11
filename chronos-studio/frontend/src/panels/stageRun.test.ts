import { describe, expect, it, vi } from "vitest";

import { completeStageRun } from "./stageRun";

describe("completeStageRun", () => {
  it("reloads the report, refreshes pipeline state, then notifies", async () => {
    const order: string[] = [];
    const reload = vi.fn(async () => { order.push("reload"); });
    const refresh = vi.fn(async () => { order.push("refresh"); });
    const notify = vi.fn(() => { order.push("notify"); });

    await completeStageRun(reload, refresh, notify);

    expect(order).toEqual(["reload", "refresh", "notify"]);
    expect(reload).toHaveBeenCalledTimes(1);
    expect(refresh).toHaveBeenCalledTimes(1);
    expect(notify).toHaveBeenCalledTimes(1);
  });
});
