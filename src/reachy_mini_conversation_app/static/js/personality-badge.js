/**
 * Header personality badge: avatar + "Personality" label + active profile name.
 * Lives in the app shell (index.html) so it persists across view changes.
 */

import { avatarFor } from "./constants.js";
import { prettifyProfileName, stripUserPrefix } from "./ui.js";

let rowEl = null;
let nameEl = null;
let avatarImg = null;

/** Bind setters to the static markup. Safe to call multiple times. */
export function mountPersonalityBadge(headerRoot = document) {
  const badge = headerRoot.querySelector('[data-component="personality-badge"]');
  if (!badge) return;
  rowEl = badge.closest('[data-component="personality-row"]');
  nameEl = badge.querySelector(".app-shell__personality-name");
  avatarImg = badge.querySelector(".app-shell__personality-avatar img");
}

/** Update the badge content. Pass a falsy name to keep the previous value. */
export function setPersonality(rawName) {
  if (!rawName || !nameEl || !avatarImg) return;
  nameEl.textContent = prettifyProfileName(rawName);
  avatarImg.src = avatarFor(stripUserPrefix(rawName));
}

export function setPersonalityBadgeVisible(visible) {
  if (rowEl) rowEl.hidden = !visible;
}
