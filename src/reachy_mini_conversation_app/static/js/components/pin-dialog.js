/** Settings PIN dialog. Privileged settings calls (secrets, Obsidian, vault access, backend and tool
 * configuration, personality saves) need the PIN; the first one on a robot without a PIN creates it.
 * Resolves the PIN, or null when the user cancels. */

import { h } from "../ui.js";

export const MIN_SETTINGS_PIN_LENGTH = 6;

export function askSettingsPin({ create = false, retry = false } = {}) {
  return new Promise((resolve) => {
    const returnFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const field = (name, label) =>
      h(
        "label",
        { class: "modal__field" },
        h("span", { class: "modal__label" }, label),
        h("input", {
          type: "password",
          name,
          required: true,
          minlength: String(create ? MIN_SETTINGS_PIN_LENGTH : 1),
          autocomplete: create ? "new-password" : "current-password",
          class: "modal__input",
        })
      );
    const subtitle = create
      ? `Choose a settings PIN of ${MIN_SETTINGS_PIN_LENGTH} or more characters. The robot asks for it before it changes secrets, Obsidian, vault access, backend, personality, or tool settings. To reset it, remove REACHY_MINI_SETTINGS_PIN_HASH from the app .env file and restart the app.`
      : "Enter the settings PIN to change this setting.";
    const errorBox = h("p", { class: "modal__error", role: "alert", "aria-live": "polite" });
    const form = h(
      "form",
      { class: "modal__form" },
      field("pin", "Settings PIN"),
      create ? field("confirm", "Confirm PIN") : null,
      errorBox,
      h(
        "div",
        { class: "modal__actions" },
        h("button", { type: "button", class: "btn btn--ghost", "data-action": "cancel" }, "Cancel"),
        h("button", { type: "submit", class: "btn btn--primary" }, create ? "Set PIN" : "Unlock")
      )
    );
    const overlay = h("div", { class: "modal-overlay", role: "presentation" });
    const dialog = h(
      "div",
      { class: "modal modal--confirm", role: "dialog", "aria-modal": "true", "aria-labelledby": "pin-title" },
      h("h2", { id: "pin-title", class: "modal__title" }, create ? "Set a settings PIN" : "Settings PIN"),
      h("p", { class: "modal__subtitle" }, subtitle),
      form
    );
    overlay.appendChild(dialog);
    document.body.appendChild(overlay);

    let settled = false;
    function close(value) {
      if (settled) return;
      settled = true;
      window.removeEventListener("keydown", onKeydown);
      overlay.remove();
      if (returnFocus?.isConnected) returnFocus.focus();
      resolve(value);
    }
    function onKeydown(event) {
      if (event.key === "Escape") close(null);
    }
    function showError(message) {
      errorBox.textContent = message;
      errorBox.classList.add("is-visible");
    }

    if (retry) showError("Wrong PIN. Try again.");
    form.addEventListener("input", () => errorBox.classList.remove("is-visible"));
    form.querySelector("[data-action='cancel']").addEventListener("click", () => close(null));
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      const data = new FormData(form);
      const pin = String(data.get("pin") || "");
      if (create && pin.length < MIN_SETTINGS_PIN_LENGTH) {
        return showError(`Use ${MIN_SETTINGS_PIN_LENGTH} or more characters.`);
      }
      if (create && pin !== String(data.get("confirm") || "")) return showError("The PINs are not the same.");
      close(pin);
    });
    window.addEventListener("keydown", onKeydown);
    requestAnimationFrame(() => form.querySelector("input")?.focus());
  });
}
