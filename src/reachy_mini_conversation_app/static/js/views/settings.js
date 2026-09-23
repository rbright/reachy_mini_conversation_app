/** Settings view: provider connection, voice, Obsidian Sync, vault access, and runtime status. */

import {
  applyVoice,
  configureObsidian,
  describeError,
  getCurrentVoice,
  getObsidianStatus,
  getStatus,
  listObsidianVaults,
  listVoices,
  obsidianLogin,
  obsidianLogout,
  saveBackendConfig,
  untilReady,
} from "../api.js";
import { BACKENDS } from "../constants.js";
import { buildProfileVaultAccessSection } from "../components/profile-vault-access.js";
import { h } from "../ui.js";

const HF_CONNECTION_MODES = Object.freeze({
  DEPLOYED: "deployed",
  LOCAL: "local",
});

const DEFAULT_HF_HOST = "localhost";
const DEFAULT_HF_PORT = 8765;
const BACKEND_LABELS = Object.freeze({
  [BACKENDS.HUGGINGFACE]: "Hugging Face",
  [BACKENDS.OPENAI]: "OpenAI Realtime",
});

const BACKEND_HINTS = Object.freeze({
  [BACKENDS.HUGGINGFACE]: "Choose the hosted service or a local realtime backend.",
  [BACKENDS.OPENAI]: "Connects directly to OpenAI with your API key.",
});

const HF_MODE_HINTS = Object.freeze({
  [HF_CONNECTION_MODES.DEPLOYED]: "Uses the hosted Hugging Face backend. No API key required.",
  [HF_CONNECTION_MODES.LOCAL]: "Connects directly to the host and port below.",
});

const OBSIDIAN_STATE_LABELS = Object.freeze({
  stopped: "Stopped",
  starting: "Starting…",
  syncing: "Syncing",
  error: "Error",
});

export async function mountSettingsView({ outlet, signal }) {
  const connectionSection = buildConnectionSection({
    onSaved: () =>
      Promise.all([
        refreshStatus({ statusSection, connectionSection, signal }),
        refreshVoices({ voiceSection, signal }),
      ]),
  });
  const voiceSection = buildVoiceSection();
  const obsidianSection = buildObsidianSection({ signal });
  const vaultAccessSection = buildProfileVaultAccessSection({ signal });
  const statusSection = buildStatusSection();

  const view = h(
    "section",
    { class: "view view--settings" },
    h(
      "header",
      { class: "view-header" },
      h("h1", { class: "view-title" }, "Settings"),
      h("p", { class: "view-subtitle" }, "Connection, voice, Obsidian Sync, and runtime state for Reachy Mini.")
    ),
    connectionSection.element,
    voiceSection.element,
    obsidianSection.element,
    vaultAccessSection.element,
    statusSection.element
  );
  outlet.replaceChildren(view);

  await Promise.all([
    refreshStatus({ statusSection, connectionSection, signal }),
    refreshVoices({ voiceSection, signal }),
    obsidianSection.refresh(),
    vaultAccessSection.refresh(),
  ]);
}

function buildConnectionSection({ onSaved } = {}) {
  const backendSelect = h(
    "select",
    { class: "settings-select", name: "backend" },
    ...Object.entries(BACKEND_LABELS).map(([value, label]) => h("option", { value }, label))
  );
  const apiKeyInput = h("input", {
    type: "password",
    name: "api_key",
    autocomplete: "new-password",
    placeholder: "Enter OpenAI API key",
    class: "settings-input",
  });
  const apiKeyField = h(
    "label",
    { class: "settings-field", "data-role": "api-key-field" },
    h("span", { class: "settings-label" }, "OpenAI API key"),
    apiKeyInput
  );
  const openaiModelSelect = h(
    "select",
    { class: "settings-select", name: "openai_model" },
    h("option", { value: "" }, "Loading models…")
  );
  const openaiModelField = h(
    "label",
    { class: "settings-field", "data-role": "openai-model-field" },
    h("span", { class: "settings-label" }, "OpenAI Realtime model"),
    openaiModelSelect
  );
  const hfModeSelect = h(
    "select",
    { class: "settings-select", name: "hf_mode" },
    h("option", { value: HF_CONNECTION_MODES.DEPLOYED }, "Hosted"),
    h("option", { value: HF_CONNECTION_MODES.LOCAL }, "Local")
  );
  const hfHostInput = h("input", {
    type: "text",
    name: "hf_host",
    autocomplete: "off",
    placeholder: DEFAULT_HF_HOST,
    value: DEFAULT_HF_HOST,
    class: "settings-input",
  });
  const hfPortInput = h("input", {
    type: "number",
    name: "hf_port",
    min: "1",
    max: "65535",
    step: "1",
    inputmode: "numeric",
    value: String(DEFAULT_HF_PORT),
    class: "settings-input",
  });
  const hfModeField = h(
    "label",
    { class: "settings-field", "data-role": "hf-mode-field" },
    h("span", { class: "settings-label" }, "Hugging Face connection"),
    hfModeSelect
  );
  const hfLocalFields = h(
    "div",
    { class: "settings-field-row", "data-role": "hf-local-fields" },
    h("label", { class: "settings-field" }, h("span", { class: "settings-label" }, "Host/IP"), hfHostInput),
    h("label", { class: "settings-field" }, h("span", { class: "settings-label" }, "Port"), hfPortInput)
  );
  const hint = h("p", { class: "settings-hint" }, "");
  const status = h("p", { class: "settings-status", role: "status", "aria-live": "polite" });
  const submitButton = h("button", { type: "submit", class: "btn btn--primary" }, "Save provider");
  let hasOpenAIKey = false;

  const form = h(
    "form",
    { class: "settings-form" },
    h("label", { class: "settings-field" }, h("span", { class: "settings-label" }, "Provider"), backendSelect),
    apiKeyField,
    openaiModelField,
    hfModeField,
    hfLocalFields,
    hint,
    h("div", { class: "settings-actions" }, submitButton),
    status
  );
  const element = h(
    "section",
    { class: "settings-section" },
    h("h2", { class: "settings-section-title" }, "Provider"),
    form
  );

  function syncProviderFields() {
    const isOpenAI = backendSelect.value === BACKENDS.OPENAI;
    const isHuggingFace = backendSelect.value === BACKENDS.HUGGINGFACE;
    const isLocalHuggingFace = isHuggingFace && hfModeSelect.value === HF_CONNECTION_MODES.LOCAL;

    apiKeyField.style.display = isOpenAI ? "" : "none";
    apiKeyInput.disabled = !isOpenAI;
    apiKeyInput.required = isOpenAI && !hasOpenAIKey;
    apiKeyInput.placeholder = hasOpenAIKey ? "Configured — enter a replacement" : "Enter OpenAI API key";
    openaiModelField.style.display = isOpenAI ? "" : "none";
    openaiModelSelect.disabled = !isOpenAI;
    hfModeField.style.display = isHuggingFace ? "" : "none";
    hfLocalFields.style.display = isLocalHuggingFace ? "" : "none";
    hfModeSelect.disabled = !isHuggingFace;
    hfHostInput.disabled = !isLocalHuggingFace;
    hfPortInput.disabled = !isLocalHuggingFace;
    hfHostInput.required = isLocalHuggingFace;
    hfPortInput.required = isLocalHuggingFace;
    hint.textContent =
      (isHuggingFace && HF_MODE_HINTS[hfModeSelect.value]) || BACKEND_HINTS[backendSelect.value] || "";
  }

  backendSelect.addEventListener("change", syncProviderFields);
  hfModeSelect.addEventListener("change", syncProviderFields);

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (submitButton.disabled) return;
    submitButton.disabled = true;
    form.setAttribute("aria-busy", "true");
    status.classList.remove("is-error");
    status.textContent = "Saving…";
    try {
      const payload = { backend: backendSelect.value };
      if (backendSelect.value === BACKENDS.OPENAI) {
        if (apiKeyInput.value) payload.api_key = apiKeyInput.value;
        payload.openai_model = openaiModelSelect.value;
      } else {
        payload.hf_mode = hfModeSelect.value;
        if (hfModeSelect.value === HF_CONNECTION_MODES.LOCAL) {
          payload.hf_host = hfHostInput.value.trim();
          if (hfPortInput.value) payload.hf_port = Number.parseInt(hfPortInput.value, 10);
        }
      }
      const result = await saveBackendConfig(payload);
      apiKeyInput.value = "";
      status.textContent = result?.message || "Saved.";
      await onSaved?.();
    } catch (error) {
      status.textContent = `Failed to save: ${describeError(error)}`;
      status.classList.add("is-error");
    } finally {
      submitButton.disabled = false;
      form.removeAttribute("aria-busy");
      syncProviderFields();
    }
  });

  syncProviderFields();

  return {
    element,
    syncFromStatus(payload) {
      if (payload?.backend_provider && BACKEND_LABELS[payload.backend_provider]) {
        backendSelect.value = payload.backend_provider;
      }
      hasOpenAIKey = Boolean(payload?.has_openai_key);
      const models = Array.isArray(payload?.openai_models) ? payload.openai_models : [];
      openaiModelSelect.replaceChildren(
        ...models.map((model) => h("option", { value: model }, model))
      );
      if (payload?.openai_model) openaiModelSelect.value = payload.openai_model;
      if (Object.values(HF_CONNECTION_MODES).includes(payload?.hf_connection_mode)) {
        hfModeSelect.value = payload.hf_connection_mode;
      }
      if (payload?.hf_direct_host) hfHostInput.value = payload.hf_direct_host;
      if (payload?.hf_direct_port != null) hfPortInput.value = String(payload.hf_direct_port);
      syncProviderFields();
    },
  };
}

function buildVoiceSection() {
  const select = h(
    "select",
    { class: "settings-select", name: "voice", disabled: "disabled" },
    h("option", { value: "" }, "Loading voices…")
  );
  const status = h("p", { class: "settings-status", role: "status", "aria-live": "polite" });
  const submitButton = h(
    "button",
    { type: "submit", class: "btn btn--primary", disabled: "disabled" },
    "Apply voice"
  );
  const form = h(
    "form",
    { class: "settings-form" },
    h("label", { class: "settings-field" }, h("span", { class: "settings-label" }, "Voice"), select),
    h("div", { class: "settings-actions" }, submitButton),
    status
  );

  const element = h(
    "section",
    { class: "settings-section" },
    h("h2", { class: "settings-section-title" }, "Voice"),
    form
  );

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (submitButton.disabled || !select.value) return;
    submitButton.disabled = true;
    select.disabled = true;
    form.setAttribute("aria-busy", "true");
    status.classList.remove("is-error");
    status.textContent = "Applying…";
    try {
      const result = await applyVoice(select.value);
      status.textContent = result?.status || "Voice applied.";
    } catch (error) {
      status.textContent = `Failed to apply: ${describeError(error)}`;
      status.classList.add("is-error");
    } finally {
      submitButton.disabled = !select.value;
      select.disabled = !select.value;
      form.removeAttribute("aria-busy");
    }
  });

  return {
    element,
    setOptions(voices, current) {
      select.replaceChildren();
      if (!voices.length) {
        select.appendChild(h("option", { value: "" }, "No voices available"));
        select.disabled = true;
        submitButton.disabled = true;
        status.textContent = "Voices are unavailable right now.";
        return;
      }
      for (const v of voices) {
        const opt = h("option", { value: v }, v);
        if (v === current) opt.selected = true;
        select.appendChild(opt);
      }
      select.disabled = false;
      submitButton.disabled = false;
      status.textContent = "";
    },
  };
}

function buildObsidianSection({ signal }) {
  const statusList = h("dl", { class: "settings-status-grid" }, statusRow("Obsidian Sync", "Loading…"));

  const emailInput = h("input", {
    type: "email",
    name: "obsidian_email",
    autocomplete: "username",
    placeholder: "Obsidian account email",
    class: "settings-input",
    required: "required",
  });
  const passwordInput = h("input", {
    type: "password",
    name: "obsidian_password",
    autocomplete: "current-password",
    placeholder: "Used once to sign in; not stored",
    class: "settings-input",
    required: "required",
  });
  const mfaInput = h("input", {
    type: "text",
    name: "obsidian_mfa",
    autocomplete: "one-time-code",
    inputmode: "numeric",
    placeholder: "Only if your account uses 2FA",
    class: "settings-input",
  });
  const loginButton = h("button", { type: "submit", class: "btn btn--primary" }, "Sign in");
  const logoutButton = h("button", { type: "button", class: "btn btn--ghost" }, "Sign out");
  const loginStatus = h("p", { class: "settings-status", role: "status", "aria-live": "polite" });
  const loginForm = h(
    "form",
    { class: "settings-form" },
    h("label", { class: "settings-field" }, h("span", { class: "settings-label" }, "Email"), emailInput),
    h("label", { class: "settings-field" }, h("span", { class: "settings-label" }, "Password"), passwordInput),
    h("label", { class: "settings-field" }, h("span", { class: "settings-label" }, "2FA code"), mfaInput),
    h("div", { class: "settings-actions" }, loginButton, logoutButton),
    loginStatus
  );

  const enabledInput = h("input", { type: "checkbox", name: "obsidian_enabled" });
  const binInput = h("input", {
    type: "text",
    name: "obsidian_headless_bin",
    autocomplete: "off",
    placeholder: "ob",
    class: "settings-input",
  });
  const vaultSelect = h(
    "select",
    { class: "settings-select", name: "obsidian_vault" },
    h("option", { value: "" }, "Load the vault list")
  );
  const loadVaultsButton = h("button", { type: "button", class: "btn btn--ghost" }, "Load vaults");
  const pathInput = h("input", {
    type: "text",
    name: "obsidian_path",
    autocomplete: "off",
    placeholder: "Default: <instance>/obsidian/<vault>",
    class: "settings-input",
  });
  const deviceInput = h("input", {
    type: "text",
    name: "obsidian_device_name",
    autocomplete: "off",
    placeholder: "reachy-mini",
    class: "settings-input",
  });
  const modeSelect = h("select", { class: "settings-select", name: "obsidian_mode" });
  const strategySelect = h("select", { class: "settings-select", name: "obsidian_conflict_strategy" });
  const e2eeInput = h("input", {
    type: "password",
    name: "obsidian_e2ee_password",
    autocomplete: "new-password",
    placeholder: "Enter the vault encryption password",
    class: "settings-input",
  });
  const saveButton = h("button", { type: "submit", class: "btn btn--primary" }, "Save Obsidian Sync");
  const configStatus = h("p", { class: "settings-status", role: "status", "aria-live": "polite" });
  const configForm = h(
    "form",
    { class: "settings-form" },
    h(
      "label",
      { class: "settings-tool-choice" },
      enabledInput,
      h("span", { class: "settings-tool-choice-copy" }, h("strong", {}, "Sync the vault on this robot"))
    ),
    h("label", { class: "settings-field" }, h("span", { class: "settings-label" }, "ob executable"), binInput),
    h(
      "div",
      { class: "settings-field-row" },
      h("label", { class: "settings-field" }, h("span", { class: "settings-label" }, "Remote vault"), vaultSelect),
      h("div", { class: "settings-actions" }, loadVaultsButton)
    ),
    h("label", { class: "settings-field" }, h("span", { class: "settings-label" }, "Local path"), pathInput),
    h("label", { class: "settings-field" }, h("span", { class: "settings-label" }, "Device name"), deviceInput),
    h(
      "div",
      { class: "settings-field-row" },
      h("label", { class: "settings-field" }, h("span", { class: "settings-label" }, "Sync mode"), modeSelect),
      h(
        "label",
        { class: "settings-field" },
        h("span", { class: "settings-label" }, "Conflict strategy"),
        strategySelect
      )
    ),
    h(
      "label",
      { class: "settings-field" },
      h("span", { class: "settings-label" }, "End-to-end encryption password"),
      e2eeInput
    ),
    h("div", { class: "settings-actions" }, saveButton),
    configStatus
  );

  const element = h(
    "section",
    { class: "settings-section" },
    h("h2", { class: "settings-section-title" }, "Obsidian Sync"),
    h(
      "p",
      { class: "settings-hint" },
      "Syncs one Obsidian vault with Obsidian Headless (ob). Install obsidian-headless on the robot first."
    ),
    statusList,
    loginForm,
    configForm
  );

  function setVaultOptions(vaults, selected) {
    const names = vaults.map((vault) => vault.name);
    if (selected && !names.includes(selected)) names.unshift(selected);
    vaultSelect.replaceChildren(
      h("option", { value: "" }, names.length ? "Choose a vault" : "Load the vault list"),
      ...names.map((name) => h("option", { value: name }, name))
    );
    vaultSelect.value = selected || "";
  }

  function render(payload) {
    statusList.replaceChildren(
      statusRow("ob", payload.available ? "Installed" : "Not found", payload.available ? "ok" : "warn"),
      statusRow(
        "Account",
        payload.signed_in === true ? "Signed in" : payload.signed_in === false ? "Not signed in" : "Unknown",
        payload.signed_in === true ? "ok" : payload.signed_in === false ? "warn" : undefined
      ),
      statusRow(
        "Sync",
        payload.enabled ? OBSIDIAN_STATE_LABELS[payload.state] || payload.state : "Off",
        payload.state === "syncing" ? "ok" : payload.state === "error" ? "warn" : undefined
      ),
      statusRow("Vault", payload.vault || "-"),
      statusRow("Local path", payload.path || "-"),
      statusRow("Last sync", payload.last_sync || "-"),
      statusRow("Encryption password", payload.has_e2ee_password ? "Configured" : "Not set")
    );
    if (payload.last_error) statusList.appendChild(statusRow("Last error", payload.last_error, "warn"));

    enabledInput.checked = Boolean(payload.enabled);
    binInput.value = payload.headless_bin || "";
    const knownVaults = Array.from(vaultSelect.options, (option) => option.value);
    if (!knownVaults.includes(payload.vault || "")) setVaultOptions([], payload.vault);
    pathInput.value = payload.custom_path || "";
    deviceInput.value = payload.device_name || "";
    modeSelect.replaceChildren(...(payload.modes || []).map((mode) => h("option", { value: mode }, mode)));
    modeSelect.value = payload.mode || "";
    strategySelect.replaceChildren(
      ...(payload.conflict_strategies || []).map((strategy) => h("option", { value: strategy }, strategy))
    );
    strategySelect.value = payload.conflict_strategy || "";
    e2eeInput.placeholder = payload.has_e2ee_password
      ? "Configured — enter a replacement"
      : "Enter the vault encryption password";
  }

  async function runAction(button, statusElement, busyText, action) {
    if (button.disabled) return;
    button.disabled = true;
    statusElement.classList.remove("is-error");
    statusElement.textContent = busyText;
    try {
      const result = await action();
      statusElement.textContent = result?.message || "Done.";
      if (result && "state" in result) render(result);
    } catch (error) {
      statusElement.textContent = `Failed: ${describeError(error)}`;
      statusElement.classList.add("is-error");
    } finally {
      button.disabled = false;
    }
  }

  loginForm.addEventListener("submit", (event) => {
    event.preventDefault();
    runAction(loginButton, loginStatus, "Signing in…", async () => {
      const payload = { email: emailInput.value.trim(), password: passwordInput.value };
      if (mfaInput.value.trim()) payload.mfa_code = mfaInput.value.trim();
      try {
        return await obsidianLogin(payload);
      } finally {
        passwordInput.value = "";
        mfaInput.value = "";
      }
    });
  });

  logoutButton.addEventListener("click", () => {
    runAction(logoutButton, loginStatus, "Signing out…", obsidianLogout);
  });

  loadVaultsButton.addEventListener("click", () => {
    runAction(loadVaultsButton, configStatus, "Loading vaults…", async () => {
      const result = await listObsidianVaults();
      const vaults = Array.isArray(result?.vaults) ? result.vaults : [];
      setVaultOptions(vaults, vaultSelect.value);
      return { message: vaults.length ? `${vaults.length} vault(s) found.` : "No remote vaults found." };
    });
  });

  configForm.addEventListener("submit", (event) => {
    event.preventDefault();
    runAction(saveButton, configStatus, "Saving…", async () => {
      const payload = {
        enabled: enabledInput.checked,
        headless_bin: binInput.value.trim(),
        vault: vaultSelect.value,
        path: pathInput.value.trim(),
        device_name: deviceInput.value.trim(),
        mode: modeSelect.value,
        conflict_strategy: strategySelect.value,
      };
      if (e2eeInput.value) payload.e2ee_password = e2eeInput.value;
      try {
        return await configureObsidian(payload);
      } finally {
        e2eeInput.value = "";
      }
    });
  });

  return {
    element,
    async refresh() {
      try {
        const payload = await untilReady(getObsidianStatus, signal);
        if (signal.aborted) return;
        render(payload);
      } catch (error) {
        if (signal.aborted) return;
        statusList.replaceChildren(statusRow("Obsidian Sync", `Unavailable: ${describeError(error)}`, "warn"));
      }
    },
  };
}

function buildStatusSection() {
  const list = h("dl", { class: "settings-status-grid" }, statusRow("Provider", "Loading…"));
  const element = h(
    "section",
    { class: "settings-section" },
    h("h2", { class: "settings-section-title" }, "Current state"),
    list
  );

  return {
    element,
    render(payload) {
      list.replaceChildren();
      const provider = payload.backend_provider;
      list.appendChild(statusRow("Provider", BACKEND_LABELS[provider] || provider || "-"));
      if (provider === BACKENDS.OPENAI) {
        list.appendChild(statusRow("Model", payload.openai_model || "-"));
        list.appendChild(
          statusRow("API key", payload.has_openai_key ? "Configured" : "Missing", payload.has_openai_key ? "ok" : "warn")
        );
      } else {
        list.appendChild(statusRow("HF connection", formatHfMode(payload.hf_connection_mode)));
        if (payload.hf_connection_mode === HF_CONNECTION_MODES.LOCAL) {
          list.appendChild(statusRow("HF target", formatHfTarget(payload)));
        }
        list.appendChild(
          statusRow("Configuration", payload.has_hf_connection ? "Ready" : "Missing", payload.has_hf_connection ? "ok" : "warn")
        );
      }
      const backendState = payload.backend_connected
        ? "connected"
        : payload.backend_connection_state || "not_started";
      const backendLabels = {
        connected: "Connected",
        connecting: "Connecting…",
        disconnected: "Disconnected",
        not_started: "Not started",
        sleeping: "Sleeping — waiting for wake word",
        restart_required: "Restart required",
        waiting_for_config: "Waiting for configuration",
      };
      list.appendChild(
        statusRow(
          "Backend",
          backendLabels[backendState] || "Unavailable",
          backendState === "connected" || backendState === "sleeping"
            ? "ok"
            : backendState === "not_started"
              ? undefined
              : "warn"
        )
      );
      if (payload.backend_error) list.appendChild(statusRow("Backend error", payload.backend_error, "warn"));
      if (payload.requires_restart) list.appendChild(statusRow("Restart", "Required to apply changes", "warn"));
    },
    renderUnavailable(error) {
      list.replaceChildren(statusRow("Backend", `Unavailable: ${describeError(error)}`, "warn"));
    },
  };
}

function statusRow(label, value, tone) {
  return h(
    "div",
    { class: ["settings-status-row", tone && `is-${tone}`] },
    h("dt", { class: "settings-status-label" }, label),
    h("dd", { class: "settings-status-value" }, value)
  );
}

function formatHfMode(mode) {
  if (mode === HF_CONNECTION_MODES.LOCAL) return "Local";
  if (mode === HF_CONNECTION_MODES.DEPLOYED) return "Hosted";
  return "-";
}

function formatHfTarget(payload) {
  const host = payload?.hf_direct_host;
  const port = payload?.hf_direct_port;
  if (!host) return "-";
  return `${host}:${port || DEFAULT_HF_PORT}`;
}

async function refreshStatus({ statusSection, connectionSection, signal }) {
  try {
    const payload = await untilReady(getStatus, signal);
    if (signal.aborted) return;
    statusSection.render(payload);
    connectionSection.syncFromStatus(payload);
  } catch (error) {
    if (signal.aborted) return;
    statusSection.renderUnavailable(error);
  }
}

async function refreshVoices({ voiceSection, signal }) {
  let voices = [];
  let current = "";
  try {
    voices = await untilReady(listVoices, signal);
  } catch {
    voices = [];
  }
  if (signal.aborted) return;
  try {
    const data = await getCurrentVoice();
    current = data?.voice || "";
  } catch {
    current = "";
  }
  if (signal.aborted) return;
  voiceSection.setOptions(voices, current);
}
