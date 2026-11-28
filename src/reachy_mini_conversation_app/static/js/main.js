/** Bootstrap: wire the shell header to the hash router and mount views on demand. */

import { ROUTES } from "./constants.js";
import { createRouter } from "./router.js";
import { mountPersonalityBadge, setPersonalityBadgeVisible } from "./personality-badge.js";
import { $ } from "./ui.js";
import { mountHomeView } from "./views/home.js";
import { mountTalkView } from "./views/talk.js";
import { mountSettingsView } from "./views/settings.js";
import { mountToolsView } from "./views/tools.js";

const SETTINGS_RETURN_KEY = "settings-return-route";

const routePath = (route) => route.split("?")[0];

/** Survives reloads while on /settings (sessionStorage may be unavailable when embedded). */
function readSettingsReturn() {
  try {
    const route = sessionStorage.getItem(SETTINGS_RETURN_KEY) || "";
    return [ROUTES.TALK, ROUTES.PERSONALITIES, ROUTES.TOOLS].includes(routePath(route)) ? route : ROUTES.TALK;
  } catch {
    return ROUTES.TALK;
  }
}

function storeSettingsReturn(route) {
  try {
    sessionStorage.setItem(SETTINGS_RETURN_KEY, route);
  } catch {
    // in-memory fallback is enough
  }
}

/** Honor the desktop app's ?theme=dark|light, overriding prefers-color-scheme. */
function applyEmbeddedTheme() {
  const theme = new URLSearchParams(window.location.search).get("theme");
  if (theme !== "dark" && theme !== "light") return;
  document.documentElement.dataset.theme = theme;
  document.documentElement.style.colorScheme = theme;
}

function boot() {
  applyEmbeddedTheme();

  const outlet = $("#view-outlet");
  if (!outlet) {
    console.error("#view-outlet missing from index.html");
    return;
  }

  const router = createRouter(
    {
      [ROUTES.TALK]: (ctx) => mountTalkView(ctx),
      [ROUTES.PERSONALITIES]: (ctx) => mountHomeView({ ...ctx, navigate: router.navigate }),
      [ROUTES.SETTINGS]: (ctx) => mountSettingsView(ctx),
      [ROUTES.TOOLS]: (ctx) => mountToolsView(ctx),
    },
    { fallback: ROUTES.TALK, outlet, onRouteChange: syncHeaderForRoute }
  );

  // Settings is an overlay: closing it returns to where it was opened
  let settingsReturn = readSettingsReturn();

  /** Where the header back button goes from `route`, and its accessible label. */
  function backTarget(route) {
    const path = routePath(route);
    if (path === ROUTES.SETTINGS) {
      const returnPath = routePath(settingsReturn);
      let label = "Back to conversation";
      if (returnPath === ROUTES.PERSONALITIES) label = "Back to personalities";
      if (returnPath === ROUTES.TOOLS) label = "Back to tools";
      return { route: settingsReturn, label };
    }
    if (path === ROUTES.TOOLS && new URLSearchParams(route.split("?")[1] || "").get("from") === "personalities") {
      return { route: ROUTES.PERSONALITIES, label: "Back to personalities" };
    }
    return { route: ROUTES.TALK, label: "Back to conversation" };
  }

  const tools = $('[data-action="open-tools"]');
  if (tools) {
    tools.addEventListener("click", (event) => {
      event.preventDefault();
      router.navigate(ROUTES.TOOLS);
    });
  }
  const gear = $('[data-action="open-settings"]');
  if (gear) {
    gear.addEventListener("click", () => {
      const route = router.currentRoute() || ROUTES.TALK;
      if (routePath(route) === ROUTES.SETTINGS) {
        router.navigate(settingsReturn);
      } else {
        settingsReturn = route;
        storeSettingsReturn(settingsReturn);
        router.navigate(ROUTES.SETTINGS);
      }
    });
  }

  const brand = $('[data-action="go-home"]');
  if (brand) {
    brand.addEventListener("click", (event) => {
      event.preventDefault();
      router.navigate(ROUTES.TALK);
    });
  }

  const personalityBadge = $('[data-action="open-personalities"]');
  if (personalityBadge) {
    personalityBadge.addEventListener("click", () => router.navigate(ROUTES.PERSONALITIES));
  }

  const back = $('[data-action="go-back"]');
  if (back) {
    back.addEventListener("click", () => router.navigate(backTarget(router.currentRoute() || ROUTES.TALK).route));
  }

  mountPersonalityBadge(document);

  function syncHeaderForRoute(route = router.currentRoute() || ROUTES.TALK) {
    const path = routePath(route);
    if (tools) {
      const onTools = path === ROUTES.TOOLS;
      tools.classList.toggle("is-active", onTools);
      if (onTools) tools.setAttribute("aria-current", "page");
      else tools.removeAttribute("aria-current");
    }
    if (gear) {
      const onSettings = path === ROUTES.SETTINGS;
      gear.classList.toggle("is-active", onSettings);
      gear.setAttribute("aria-label", onSettings ? "Close settings" : "Open settings");
    }
    if (back) {
      back.hidden = path === ROUTES.TALK;
      back.setAttribute("aria-label", backTarget(route).label);
    }
    setPersonalityBadgeVisible(path === ROUTES.TALK);
  }
  router.start();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", boot, { once: true });
} else {
  boot();
}
