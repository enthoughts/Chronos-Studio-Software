/**
 * Finish a stage-marking POST in the same order every panel needs:
 * reload the panel's stored report, refresh the pipeline state, then notify.
 * Without the refresh, the backend can be complete while the stepper still
 * says "Not started" and downstream stages stay gated.
 */
export async function completeStageRun(
  reload: () => Promise<void>,
  refresh: () => Promise<void>,
  notify: () => void,
): Promise<void> {
  await reload();
  await refresh();
  notify();
}
