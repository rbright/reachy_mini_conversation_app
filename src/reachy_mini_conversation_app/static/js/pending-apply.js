/**
 * Cross-view hand-off for an in-flight personality apply: set by home.js
 * before navigating to /talk, consumed by talk.js to drive its caption.
 */

let pending = null;

/**
 * Record an in-flight personality apply, or clear it with null.
 *
 * @param {{ name: string, promise: Promise<unknown> } | null} entry
 */
export function setPendingApply(entry) {
  pending = entry;
  // Avoid an unhandledrejection if /talk never mounts to consume this.
  entry?.promise.catch(() => {});
}

/**
 * Read and clear the in-flight apply, if any. Single-shot by design:
 * the consumer (talk view) owns the lifecycle from here on.
 *
 * @returns {{ name: string, promise: Promise<unknown> } | null}
 */
export function consumePendingApply() {
  const entry = pending;
  pending = null;
  return entry;
}
