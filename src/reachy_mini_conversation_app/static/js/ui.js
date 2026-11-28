/** Tiny DOM and display-name helpers. */
export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value == null || value === false) continue;
    if (key === "class") {
      el.className = Array.isArray(value) ? value.filter(Boolean).join(" ") : String(value);
    } else if (key === "style" && typeof value === "object") {
      Object.assign(el.style, value);
    } else if (key === "dataset" && typeof value === "object") {
      for (const [dk, dv] of Object.entries(value)) {
        if (dv != null) el.dataset[dk] = String(dv);
      }
    } else if (key.startsWith("on") && typeof value === "function") {
      el.addEventListener(key.slice(2).toLowerCase(), value);
    } else if (key === "html") {
      // Escape hatch for trusted HTML (e.g. inline SVG icons we author).
      el.innerHTML = String(value);
    } else {
      el.setAttribute(key, String(value));
    }
  }
  for (const child of children.flat(Infinity)) {
    if (child == null || child === false || child === true) continue;
    el.appendChild(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return el;
}

export function $(selector, root = document) {
  return root.querySelector(selector);
}

/** Labelled settings control: `<label class="settings-field">` with its caption. */
export function settingsField(label, control) {
  return h("label", { class: "settings-field" }, h("span", { class: "settings-label" }, label), control);
}

/** A fragment of `<option>`s for a personality picker payload ({ profiles, profile }). */
export function profileOptions(payload) {
  const options = document.createDocumentFragment();
  for (const profile of payload.profiles || []) {
    options.append(
      h(
        "option",
        { value: profile.id, selected: profile.id === payload.profile ? "selected" : null },
        `${prettifyProfileName(profile.id)}${profile.active ? " · Active" : ""}`
      )
    );
  }
  return options;
}

export function stripUserPrefix(name) {
  return name.replace(/^user_personalities\//, "");
}

export function prettifyProfileName(name) {
  return stripUserPrefix(name)
    .split(/[_-]/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

export function prettifyToolName(toolId) {
  return prettifyProfileName(toolId.split("__").at(-1));
}
