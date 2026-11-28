/**
 * Talk view: conversation orb driven by the RPC activity stream.
 * Audio I/O runs entirely in Python; the orb doubles as the mic toggle.
 * Robot stays live, tapping the orb only mutes or unmutes the user's mic.
 */

import { applyPersonality, getMicState, listPersonalities, setMicMuted, subscribe } from "../api.js";
import { createConversationPanel } from "../components/conversation-panel.js";
import { ORB_STATES } from "../constants.js";
import { createOrb, mapActivityToState } from "../orb.js";
import { consumePendingApply } from "../pending-apply.js";
import { setPersonality } from "../personality-badge.js";
import { h, prettifyProfileName } from "../ui.js";

const CAPTION_BY_STATE = Object.freeze({
  [ORB_STATES.MUTED]: "Muted",
  [ORB_STATES.IDLE]: "Ready",
  [ORB_STATES.CONNECTING]: "Connecting to the backend...",
  [ORB_STATES.LISTENING]: "Listening",
  [ORB_STATES.THINKING]: "Thinking",
  [ORB_STATES.SPEAKING]: "Speaking",
  [ORB_STATES.ERROR]: "Connection error",
});

export async function mountTalkView({ outlet, signal }) {
  const pending = consumePendingApply();
  const micStatePromise = getMicState().catch((error) => {
    console.warn("Failed to load microphone state", error);
    return null;
  });
  let muted = false;
  let micReady = false;
  let togglePending = false;
  let activePersonality = null;
  let unsubscribeActivity = null;

  const caption = h(
    "p",
    { class: "talk__caption", role: "status", "aria-live": "polite" },
    CAPTION_BY_STATE[ORB_STATES.CONNECTING]
  );
  const defaultAction = document.querySelector('[data-component="default-personality-action"]');
  if (defaultAction) {
    defaultAction.hidden = true;
    defaultAction.addEventListener("click", onSetDefault);
  }
  const orb = createOrb({
    initialState: ORB_STATES.CONNECTING,
    onStateChange: (state) => {
      caption.textContent = CAPTION_BY_STATE[state] || "";
    },
  });
  orb.root.disabled = true;
  orb.root.addEventListener("click", onMicTap);
  syncMicAria();
  const conversationPanel = createConversationPanel({
    toggle: document.querySelector('[data-component="conversation-toggle"]'),
  });

  signal.addEventListener("abort", cleanup, { once: true });

  const view = h(
    "section",
    { class: "view view--talk" },
    h("div", { class: "talk__stage" }, h("div", { class: "talk__orb-wrap" }, orb.root), caption),
    conversationPanel.root
  );
  outlet.replaceChildren(view);

  if (pending) {
    caption.textContent = `Applying "${prettifyProfileName(pending.name)}"…`;
    try {
      await pending.promise;
    } catch (error) {
      if (signal.aborted) return;
      orb.setState(ORB_STATES.ERROR);
      caption.textContent = `Failed to apply personality: ${error?.message || error}`;
      return;
    }
    if (signal.aborted) return;
    // The activity subscription will flip the orb to its resting state next tick.
    caption.textContent = CAPTION_BY_STATE[ORB_STATES.CONNECTING];
  }
  // Also covers a deep link to /talk with no pending apply: the header badge needs the active personality.
  void refreshPersonalityState();

  const micState = await micStatePromise;
  if (micState) muted = Boolean(micState.muted);
  if (signal.aborted) return;
  micReady = true;
  orb.root.disabled = false;
  syncMicAria();

  // The shared /rpc client (api.js) owns reconnection.
  unsubscribeActivity = subscribe("conversation.activity", (params) => {
    if (muted) return;
    const next = mapActivityToState((params?.reason || "").trim());
    if (next != null) orb.setState(next);
  });

  // Re-sync mic state after subscribing: another tab may have toggled it.
  if (!togglePending) {
    try {
      muted = Boolean((await getMicState())?.muted);
    } catch (error) {
      console.warn("Failed to refresh microphone state; keeping the last known state", error);
    }
  }
  if (signal.aborted) return;
  showRestingState();

  function cleanup() {
    unsubscribeActivity?.();
    conversationPanel.dispose();
    orb.dispose();
    if (defaultAction) {
      defaultAction.hidden = true;
      defaultAction.removeEventListener("click", onSetDefault);
    }
  }

  function showRestingState() {
    const state = muted ? ORB_STATES.MUTED : ORB_STATES.IDLE;
    orb.setState(state);
    // setState skips unchanged states, so set the caption explicitly
    caption.textContent = CAPTION_BY_STATE[state];
    syncMicAria();
  }

  async function onMicTap() {
    if (!micReady || togglePending) return;
    togglePending = true;
    try {
      const data = await setMicMuted(!muted);
      muted = Boolean(data?.muted);
    } catch (error) {
      if (!signal.aborted) {
        caption.textContent = `Failed to toggle the microphone: ${error?.message || error}`;
      }
      return;
    } finally {
      togglePending = false;
    }
    if (signal.aborted) return;
    showRestingState();
  }

  async function refreshPersonalityState() {
    let data;
    try {
      data = await listPersonalities();
    } catch (error) {
      console.warn("Failed to load the active personality", error);
      return;
    }
    if (signal.aborted || !data?.current) return;
    activePersonality = data.current;
    setPersonality(data.current);
    if (defaultAction) defaultAction.hidden = Boolean(data.locked) || data.current === (data.startup || "default");
  }

  async function onSetDefault() {
    if (!defaultAction || !activePersonality) return;
    defaultAction.disabled = true;
    caption.textContent = `Saving "${prettifyProfileName(activePersonality)}" as default...`;
    try {
      await applyPersonality(activePersonality, { persist: true });
      if (signal.aborted) return;
      defaultAction.hidden = true;
      caption.textContent = `"${prettifyProfileName(activePersonality)}" will be used at startup.`;
    } catch (error) {
      if (!signal.aborted) {
        caption.textContent = `Failed to save default: ${error?.message || error}`;
      }
    } finally {
      defaultAction.disabled = false;
    }
  }

  function syncMicAria() {
    if (!micReady) {
      orb.root.setAttribute("aria-pressed", "false");
      orb.root.setAttribute("aria-label", "Loading microphone state");
      return;
    }
    orb.root.setAttribute("aria-pressed", String(!muted));
    orb.root.setAttribute("aria-label", muted ? "Unmute microphone" : "Mute microphone");
  }
}
