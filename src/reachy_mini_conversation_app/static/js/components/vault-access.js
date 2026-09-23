/** Per-personality Obsidian vault access controls. */

import { describeError, getProfileVaultAccess, saveProfileVaultAccess } from "../api.js";
import { h, prettifyProfileName } from "../ui.js";

const WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];

export function buildProfileVaultAccessSection({ signal } = {}) {
  const profileSelect = h("select", {
    class: "settings-select",
    name: "vault_access_profile",
    disabled: "disabled",
    "aria-label": "Personality to configure",
  });
  const agentInput = textInput("vault_agent", "Agent id in the vault schema, for example emma");
  const readInput = linesInput("vault_read", "One vault folder per line");
  const writeInput = linesInput("vault_write", "One vault folder per line");
  const contextInput = linesInput("vault_session_context", "One note path per line, read at session start");
  const logFolderInput = textInput("vault_log_folder", "Folder for one note per session");
  const logTypeInput = textInput("vault_log_type", "Note type from System/Schema.md");
  const logPropertiesInput = linesInput("vault_log_properties", "key: {date}  (one property per line)");
  const memoryFolderInput = textInput("vault_memory_folder", "Folder for one note per week");
  const memoryTypeInput = textInput("vault_memory_type", "Note type from System/Schema.md");
  const memoryWeekdaySelect = h(
    "select",
    { class: "settings-select", name: "vault_memory_weekday" },
    ...WEEKDAYS.map((day, index) => h("option", { value: String(index + 1) }, day))
  );
  const memoryPropertiesInput = linesInput("vault_memory_properties", "key: {week_start}  (one property per line)");
  const status = h("p", { class: "settings-status", role: "status", "aria-live": "polite" });
  const removeButton = h("button", { type: "button", class: "btn btn--ghost", disabled: "disabled" }, "Remove access");
  const saveButton = h("button", { type: "submit", class: "btn btn--primary", disabled: "disabled" }, "Save vault access");
  const form = h(
    "form",
    { class: "settings-form" },
    field("Configure for", profileSelect),
    field("Agent", agentInput),
    h("div", { class: "settings-field-row" }, field("Read folders", readInput), field("Write folders", writeInput)),
    field("Session context notes", contextInput),
    h("h3", { class: "settings-label" }, "Session log"),
    h("div", { class: "settings-field-row" }, field("Folder", logFolderInput), field("Type", logTypeInput)),
    field("Properties", logPropertiesInput),
    h("h3", { class: "settings-label" }, "Weekly memory"),
    h("div", { class: "settings-field-row" }, field("Folder", memoryFolderInput), field("Type", memoryTypeInput)),
    field("{date} is this day of the summarized week", memoryWeekdaySelect),
    field("Properties", memoryPropertiesInput),
    h("div", { class: "settings-actions" }, removeButton, saveButton),
    status
  );
  const element = h(
    "section",
    { class: "settings-section" },
    h("h2", { class: "settings-section-title" }, "Vault access"),
    h(
      "p",
      { class: "settings-hint settings-section-intro" },
      "Folders each personality may read and write in the synced vault with the vault tools; . is the vault root. " +
        "The vault's System/Schema.md must also allow them. Property placeholders: {date} {time} {slug} {title} " +
        "{week} {month} {year} {quarter}; weekly memory also {week_start} {week_end}."
    ),
    form
  );

  let current = null;
  let busy = false;

  function syncActions() {
    const editable = current?.editable !== false;
    profileSelect.disabled = busy || !current?.profiles?.length;
    saveButton.disabled = busy || !editable || !current;
    removeButton.disabled = busy || !editable || !current?.access;
  }

  function render(payload) {
    current = payload;
    profileSelect.replaceChildren(
      ...(payload.profiles || []).map((profile) =>
        h(
          "option",
          { value: profile.id, selected: profile.id === payload.profile ? "selected" : null },
          `${prettifyProfileName(profile.id)}${profile.active ? " · Active" : ""}`
        )
      )
    );
    const access = payload.access || {};
    agentInput.value = access.agent || "";
    readInput.value = (access.read || []).join("\n");
    writeInput.value = (access.write || []).join("\n");
    contextInput.value = (access.session_context || []).join("\n");
    logFolderInput.value = access.session_log?.folder || "";
    logTypeInput.value = access.session_log?.type || "";
    logPropertiesInput.value = formatProperties(access.session_log?.properties);
    memoryFolderInput.value = access.weekly_memory?.folder || "";
    memoryTypeInput.value = access.weekly_memory?.type || "";
    memoryWeekdaySelect.value = String(access.weekly_memory?.date_weekday || 1);
    memoryPropertiesInput.value = formatProperties(access.weekly_memory?.properties);
    syncActions();
  }

  function collectAccess() {
    const access = {
      agent: agentInput.value.trim(),
      read: lines(readInput.value),
      write: lines(writeInput.value),
      session_context: lines(contextInput.value),
    };
    if (logFolderInput.value.trim() || logTypeInput.value.trim()) {
      access.session_log = {
        folder: logFolderInput.value.trim(),
        type: logTypeInput.value.trim(),
        properties: parseProperties(logPropertiesInput.value, "Session log"),
      };
    }
    if (memoryFolderInput.value.trim() || memoryTypeInput.value.trim()) {
      access.weekly_memory = {
        folder: memoryFolderInput.value.trim(),
        type: memoryTypeInput.value.trim(),
        date_weekday: Number(memoryWeekdaySelect.value),
        properties: parseProperties(memoryPropertiesInput.value, "Weekly memory"),
      };
    }
    return access;
  }

  async function run(message, action) {
    busy = true;
    syncActions();
    status.classList.remove("is-error");
    status.textContent = message;
    try {
      const payload = await action();
      if (signal?.aborted) return;
      render(payload);
      status.textContent = payload?.message || "";
    } catch (error) {
      if (signal?.aborted) return;
      status.textContent = describeError(error);
      status.classList.add("is-error");
    } finally {
      busy = false;
      if (!signal?.aborted) syncActions();
    }
  }

  profileSelect.addEventListener("change", () =>
    run("Loading vault access…", () => getProfileVaultAccess(profileSelect.value))
  );
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    if (!current) return;
    run("Saving vault access…", () => saveProfileVaultAccess(current.profile, collectAccess()));
  });
  removeButton.addEventListener("click", () => {
    if (!current) return;
    run("Removing vault access…", () => saveProfileVaultAccess(current.profile, null));
  });

  return {
    element,
    refresh() {
      return run("Loading vault access…", () => getProfileVaultAccess(current?.profile));
    },
  };
}

function field(label, control) {
  return h("label", { class: "settings-field" }, h("span", { class: "settings-label" }, label), control);
}

function textInput(name, placeholder) {
  return h("input", { type: "text", name, placeholder, class: "settings-input", autocomplete: "off" });
}

function linesInput(name, placeholder) {
  return h("textarea", { name, placeholder, rows: "3", class: "settings-input" });
}

function lines(text) {
  return text
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
}

function formatProperties(properties) {
  return Object.entries(properties || {})
    .map(([key, value]) => `${key}: ${value}`)
    .join("\n");
}

/** Parse `key: value` lines; throw on a line without a key so that the save stops with a message. */
function parseProperties(text, label) {
  const properties = {};
  const invalid = [];
  for (const line of lines(text)) {
    const separator = line.indexOf(":");
    if (separator > 0) properties[line.slice(0, separator).trim()] = line.slice(separator + 1).trim();
    else invalid.push(line);
  }
  if (invalid.length) {
    throw new Error(`${label} properties need one "key: value" per line. Fix: ${invalid.join(" | ")}`);
  }
  return properties;
}
