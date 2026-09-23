---
title: Reachy Mini Conversation App
emoji: 🎤
colorFrom: red
colorTo: blue
sdk: static
pinned: false
short_description: Talk with Reachy Mini!
suggested_storage: large
tags:
 - reachy_mini
 - reachy_mini_python_app
---

# Reachy Mini conversation app

Conversational app for the Reachy Mini robot combining realtime voice, vision, personality-aware tools, and choreographed motion.

![Reachy Mini Dance](docs/assets/reachy_mini_dance.gif)

## Table of contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the app](#running-the-app)
- [LLM tools](#llm-tools-exposed-to-the-assistant)
- [Creating and adding tools](#creating-and-adding-tools)
- [Advanced features](#advanced-features)
- [Contributing](#contributing)
- [License](#license)

## Overview

- Low-latency audio conversation through either the supported Hugging Face backend or the direct OpenAI Realtime API.
- Vision is handled by the realtime backend when the `camera` tool is used.
- Layered motion system queues primary moves (dances, emotions, goto poses, breathing) while blending speech-reactive wobble.
- Async tools integrate motion, camera capture, and MCP Tool Spaces. The optional web UI (`--ui`) manages conversations, personalities, tools, and settings.

## Architecture

The app connects the user, AI services, and robot hardware:

<p align="center">
  <img src="docs/assets/conversation_app_arch.svg" alt="Architecture Diagram" width="600"/>
</p>

## Installation

> [!IMPORTANT]
> Install [Reachy Mini's SDK](https://github.com/pollen-robotics/reachy_mini/) before using this app.<br>
> Windows support is currently experimental and has not been extensively tested. Use with caution.

<details open>
<summary>Using uv (recommended)</summary>

Set up with [uv](https://docs.astral.sh/uv/):

```bash
# macOS (Homebrew)
uv venv --python /opt/homebrew/bin/python3.12 .venv

# Linux / Windows (Python in PATH)
uv venv --python python3.12 .venv

source .venv/bin/activate
uv sync
```

Include dev dependencies:
```bash
uv sync --group dev
```

</details>

> [!NOTE]
> Run `uv sync --frozen` to install the exact dependency set from `uv.lock` without re-resolving versions.

<details>
<summary>Using pip</summary>

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Install dev dependencies:
```bash
pip install -e .[dev]                   # Development tools
```

</details>

## Configuration

The default setup uses the Hugging Face backend and does not require an API key. Select OpenAI Realtime in the web UI to use direct speech-to-speech sessions with your own OpenAI API key.

Copy `.env.example` to `.env` to configure a provider outside the web UI.

| Variable | Description |
|----------|-------------|
| `BACKEND_PROVIDER` | Realtime provider: `huggingface` (default) or `openai`. The Settings UI persists this choice. |
| `OPENAI_API_KEY` | API key for direct OpenAI Realtime sessions. The Settings UI can save or replace it and never returns it to the browser. |
| `OPENAI_REALTIME_MODEL` | Direct OpenAI Realtime model. Defaults to `gpt-realtime-2.1`; supported values are shown in the Settings UI. |
| `REALTIME_TRANSCRIPTION_LANGUAGE` | Optional input transcription language for the realtime backend. Defaults to `en`; set to a backend-supported code such as `zh` for Chinese. |
| `HF_REALTIME_CONNECTION_MODE` | Hugging Face connection selector: `deployed` uses the built-in Hugging Face server; `local` uses `HF_REALTIME_WS_URL`. Defaults to `deployed`. |
| `HF_REALTIME_WS_URL` | Direct websocket endpoint for your own Hugging Face backend. Accepts either a base URL like `ws://127.0.0.1:8765/v1` or the full websocket URL `ws://127.0.0.1:8765/v1/realtime`. Used when `HF_REALTIME_CONNECTION_MODE=local`. |
| `HF_TOKEN` | Optional token for Hugging Face access. Local endpoints receive only this explicitly configured token. |
| `REACHY_MINI_APP_TIMEOUT_MINUTES` | Minutes of inactivity before Reachy goes to sleep and the app stops. Defaults to `1440` (one day); set to `0` to disable. |
| `REACHY_MINI_WAKE_WORD_ENABLED` | Enables local wake gating. Defaults to `true`; set to `false` for always-on conversations. |
| `REACHY_MINI_WAKE_WORD_MODEL` | Optional path to a custom openWakeWord ONNX model. Defaults to the bundled `Hey Emma` model. |
| `REACHY_MINI_WAKE_WORD_THRESHOLD` | Wake confidence threshold, greater than `0` through `1`. Defaults to `0.5`. |
| `REACHY_MINI_SLEEP_PHRASES` | Comma-separated phrases matched against final user transcripts. Defaults include `Goodbye Emma` and `go to sleep`. |
| `OBSIDIAN_SYNC_ENABLED` | Runs Obsidian Sync for one vault. Defaults to `false`. See [Obsidian Sync](#obsidian-sync). |
| `OBSIDIAN_HEADLESS_BIN` | Obsidian Headless executable. Defaults to `ob` on `PATH`. |
| `OBSIDIAN_SYNC_VAULT` | Remote vault name or ID. The Settings UI lists the vaults of the signed-in account. |
| `OBSIDIAN_SYNC_PATH` | Local vault folder. Defaults to `<instance path>/obsidian/<vault>`. |
| `OBSIDIAN_SYNC_DEVICE_NAME` | Device name in the vault's sync history. Defaults to `reachy-mini`. |
| `OBSIDIAN_SYNC_MODE` | `bidirectional` (default) or `pull-only`. `mirror-remote` is refused because it reverts local writes. |
| `OBSIDIAN_SYNC_CONFLICT_STRATEGY` | `merge` (default) or `conflict`. |
| `OBSIDIAN_SYNC_E2EE_PASSWORD` | End-to-end encryption password of the remote vault. The Settings UI can save or replace it and never returns it to the browser. |

With wake gating enabled, the microphone runs only the local wake-word model until it detects the wake phrase. Reachy wakes and starts a fresh realtime session. A configured sleep phrase closes that session, moves Reachy to its sleep pose, and leaves the local detector running for the next wake phrase. The optional openWakeWord runtime is installed on Linux ARM64 with Python 3.11 or 3.12, which covers the Reachy Mini deployment. On other platforms or when that runtime cannot load, the app logs the error and continues in always-on mode.

### OpenAI Realtime

OpenAI uses a direct 24 kHz Realtime WebSocket audio path. It does not use the Hugging Face speech-to-speech service. Configure it in Settings, or set:

```env
BACKEND_PROVIDER=openai
OPENAI_API_KEY=your-api-key
OPENAI_REALTIME_MODEL=gpt-realtime-2.1
```

The API key is stored only in the instance `.env`. Enter a new key in Settings to replace it. The UI reports only whether a key exists.

### Hugging Face Connection Modes

Use the built-in Hugging Face server through the app-managed Space proxy. This is the default for a new install; set it explicitly only when you want to switch back from a saved local endpoint:

```env
HF_REALTIME_CONNECTION_MODE=deployed
```

Deployed session allocation falls back to cached `hf auth login` credentials and reports the daemon-provided hardware ID when available. Cached credentials and the hardware ID are not sent to local endpoints.

Run your own realtime voice backend using [speech-to-speech](https://github.com/huggingface/speech-to-speech) on the same machine as the conversation app:

```env
HF_REALTIME_CONNECTION_MODE=local
HF_REALTIME_WS_URL=ws://127.0.0.1:8765/v1/realtime
```

Run your own Hugging Face backend on your laptop and connect to it from Reachy Mini Wireless over the same Wi-Fi network:

```env
HF_REALTIME_CONNECTION_MODE=local
HF_REALTIME_WS_URL=ws://<your-laptop-lan-ip>:8765/v1/realtime
```

For that LAN setup, make sure the backend listens on an address reachable from the robot, not only on `127.0.0.1`.

If the backend stays bound to loopback on your laptop, you can forward it into the robot over SSH instead:

```bash
ssh -N -R 8765:127.0.0.1:8765 <robot-user>@<robot-host>
```

Then set this on the robot:

```env
HF_REALTIME_CONNECTION_MODE=local
HF_REALTIME_WS_URL=ws://127.0.0.1:8765/v1/realtime
```

In the web UI's Settings view, select Hugging Face or OpenAI Realtime. Hugging Face keeps its hosted/local controls. OpenAI exposes its API key, validated model catalog, and native voice catalog. Voice choices update for the selected provider.

### Obsidian Sync

The app can keep one [Obsidian Sync](https://obsidian.md/sync) vault synced on the robot with [Obsidian Headless](https://www.npmjs.com/package/obsidian-headless) (`ob`). The app does not install `ob` or Node.js: install `obsidian-headless` (Node.js 22 or later) on the device first. If `ob` is missing, the Settings UI reports it and the conversation keeps working.

Configure it in the Settings view, in the Obsidian Sync card:

1. Sign in with your Obsidian account email, password, and 2FA code if you use one. The app passes them to `ob login` once and does not store them. `ob` keeps its own sign-in state in the app user's home folder.
2. Load the remote vaults, choose one, and set the other options. Leave the local path blank to use `<instance path>/obsidian/<vault>`.
3. Enter the vault's end-to-end encryption password if it has one, turn sync on, and save.

The app links the local folder with `ob sync-setup` when needed, applies the mode, conflict strategy, and device name with `ob sync-config`, and runs `ob sync --continuous`. It writes the `ob` output to the app log, with the encryption password redacted. If `ob sync` fails, the app restarts it after 5 s, 30 s, and 120 s, then every 5 minutes. On app stop, it sends `SIGINT` and waits 10 s before it sends `SIGTERM`.

Passwords reach `ob` on stdin, not on the command line. The encryption password is stored in the instance `.env` (mode `0600`), like the OpenAI key; the UI reports only whether it exists.

#### Vault access for personalities

The `vault_read`, `vault_query`, and `vault_write` tools use the synced vault. The vault must have a `System/Schema.md` note with one `yaml vault-schema` block (the vault contract). Enable the tools for a personality in Tools → Tool access, then set its vault access in Settings → Vault access. The app stores it in `profile_vault_access.json`, next to `profile_toolsets.json`:

```json
{
  "version": 1,
  "profiles": {
    "Emma": {
      "agent": "emma",
      "read": ["Emma/Conversation Playbook", "Emma/Sessions", "Emma/Weekly Memories"],
      "write": ["Emma/Sessions", "Emma/Weekly Memories"],
      "session_context": ["Emma/Conversation Playbook/Current.md"],
      "session_log": {"folder": "Emma/Sessions", "type": "emma-session", "properties": {"date": "{date}"}},
      "weekly_memory": {
        "folder": "Emma/Weekly Memories",
        "type": "emma-memory",
        "date_weekday": 5,
        "properties": {"date": "{date}", "week_of": "{week_start}"}
      }
    }
  }
}
```

- A folder must be in this list and in the `agents` section of the vault schema for the same agent. No agent writes `System/`.
- `vault_write` sets `created`, `author: agent/<agent>`, and `run: reachy:<agent>:<session-id>`, and checks the note type, folder, and keys against the schema. It refuses paths with `..` or symbolic links, non-Markdown files, existing logs and dated notes, `approved` or `superseded` artifacts, and text that looks like a credential. It writes a temporary file and renames it.
- At session start, the app adds the agent's section of the vault `AGENTS.md`, a schema summary, and the `session_context` notes to the instructions (8000 characters at most). Reconnects in the same wake session reuse them.
- At session end (sleep phrase, app stop, or shutdown), the app writes one `session_log` note with the transcript. It writes nothing when the user did not speak. The note name comes from the type's `name` rule in the schema. At the first session end of a new ISO week, it also writes one `weekly_memory` note that lists last week's session logs. `{date}` in a weekly memory is the `date_weekday` day of that week.

## Running the app

Activate your virtual environment, then launch:

```bash
reachy-mini-conversation-app
```

> [!TIP]
> Make sure the Reachy Mini daemon is running before launching the app. If you see a `TimeoutError`, it means the daemon isn't started. See [Reachy Mini's SDK](https://github.com/pollen-robotics/reachy_mini/) for setup instructions.

The app runs in console mode. Add `--ui` to serve the web interface at http://127.0.0.1:7860/.

### CLI options

| Option | Default | Description |
|--------|---------|-------------|
| `--no-camera` | `False` | Run without camera capture. |
| `--ui` | `False` | Serve the web UI at http://127.0.0.1:7860/, in addition to console mode. |
| `--robot-name` | `None` | Optional. Connect to a specific robot by name when running multiple daemons on the same subnet. See [Multiple robots on the same subnet](#advanced-features). |
| `--debug` | `False` | Enable verbose logging for troubleshooting. |

### Examples

```bash
# Audio-only conversation (no camera)
reachy-mini-conversation-app --no-camera

# Launch with the minimal web UI for personality/mic/settings control
reachy-mini-conversation-app --ui
```

## LLM tools exposed to the assistant

The default profile exposes these tools. Use Tools → Tool access to customize any profile.
Every bundled profile enables `head_tracking` by default; users can still disable it per personality.

| Tool | Action | Dependencies |
|------|--------|--------------|
| `dance` | Queue a dance from `reachy_mini_dances_library`. | Core install only. |
| `stop_dance` | Clear queued dances. | Core install only. |
| `play_emotion` | Play a recorded emotion clip via Hugging Face datasets. | Core install only. Uses the default open emotions dataset: [`pollen-robotics/reachy-mini-emotions-library`](https://huggingface.co/datasets/pollen-robotics/reachy-mini-emotions-library). |
| `stop_emotion` | Clear queued emotions. | Core install only. |
| `camera` | Capture the latest camera frame and analyze it with the selected realtime backend. | Core install only. Requires the camera (disable with `--no-camera`). |
| `idle_do_nothing` | Explicitly remain idle during an idle turn. Not intended for normal conversation turns. | Core install only. |
| `move_head` | Queue a head pose change (left/right/up/down/front). | Core install only. |
| `head_tracking` | Follow the user's face with the head, or stop following. | Core install only. Requires a daemon with the `vision` extra and a camera. |
| `go_to_sleep` | Run Reachy's sleep movement and stop the current app after an explicit user request. | Core install only. |
| `sweep_look` | Sweep Reachy's head left, right, and back to center. | Shared tool, enabled by default in the default profile. |
| `remember` | Save one short, stable fact about the user for future sessions. | Core install only. Stored in the app instance data directory. |
| `forget` | Remove a saved memory fact by matching a short query. | Core install only. |
| `vault_read` | Read one note from the synced Obsidian vault. | Needs [Obsidian Sync](#obsidian-sync) and vault access for the personality. |
| `vault_query` | List vault notes by folder, type, status, and created date. | Needs Obsidian Sync and vault access. |
| `vault_write` | Create a note, or update a draft or owned record, under the vault contract. | Needs Obsidian Sync and write access. |
| `volume_control` | Read or change Reachy's speaker or microphone volume. | Core install only. Uses the daemon REST API; setting the speaker volume plays a short confirmation sound. |
| `robot_status` | Read one status topic: `name`, `software` (version, update available), `wifi` (IP address, network), `account` (Hugging Face sign-in), `imu` (which way the head is tilted, motion, temperature), `apps` (installed apps). | Core install only. Uses the daemon REST API. The update check and the Wi-Fi network details are wireless-version only; the IP address is reported on any robot. |
| `pollen_robotics_reachy_mini_search_tool__search_web` | Search the web and return a short list of results. | Preinstalled MCP Space: `pollen-robotics/reachy-mini-search-tool`. |
| `pollen_robotics_reachy_mini_weather_tool__get_weather` | Report today's weather for a place: current conditions, high and low temperature, and rain chance. | Preinstalled MCP Space: `pollen-robotics/reachy-mini-weather-tool`. |
| `pollen_robotics_reachy_mini_time_tool__get_time` | Report the current time for a timezone or the user's local time, or the difference between two timezones. | Preinstalled MCP Space: `pollen-robotics/reachy-mini-time-tool`. |

> [!NOTE]
> `remember`/`forget` facts are stored in `memory.v1.json` inside the app's instance data directory (`~/.local/share/reachy_mini_conversation_app/` by default, or the instance path used by the desktop launcher). `forget` only removes facts matched by query. To reset all remembered facts, delete this file.

## Creating and adding tools

Tools can run locally as Python code or remotely in an MCP-compatible Hugging Face Space. Keep robot, camera, and local-data operations in local tools. A Space is a better fit for shareable, stateless services such as search and external API lookups.

### Local tools

Create one Python module per tool, with the file name matching the tool's unique `name`. See [`idle_do_nothing.py`](src/reachy_mini_conversation_app/tools/idle_do_nothing.py) for a minimal implementation.

Each tool subclasses `Tool` and defines `name`, a model-facing `description`, an object-shaped JSON Schema in `parameters_schema`, and an async `__call__` method. Use `ToolDependencies` for runtime services, and set `needs_response = False` for actions that should not trigger a spoken follow-up. Catch expected operational failures, log them with the module logger, and return `{"error": "..."}` so the conversation can continue.

Restart the app after adding the module. Use Tools → Tool access to enable it for a personality, or add its name to that profile's `default_tools` in `profile.md`. See [External profiles and tools](#external-profiles-and-tools) for external directories and autoload behavior.

### Hugging Face Space tools

To publish a remote tool, create a Gradio Space, expose its API as MCP with `mcp_server=True`, and give each function clear type hints and docstrings. Verify that `https://<space-subdomain>.hf.space/gradio_api/mcp/schema` lists the expected tools before installing the Space.

Use the maintained [weather](https://huggingface.co/spaces/pollen-robotics/reachy-mini-weather-tool), [time](https://huggingface.co/spaces/pollen-robotics/reachy-mini-time-tool), and [search](https://huggingface.co/spaces/pollen-robotics/reachy-mini-search-tool) Spaces as examples. See Gradio's [MCP server guide](https://www.gradio.app/guides/building-mcp-server-with-gradio) for additional publishing guidance and [Installing Hugging Face Space tools](#installing-hugging-face-space-tools) for this app's installation steps.

## Advanced features

Built-in motion content is published as open Hugging Face datasets:

- Emotions: [`pollen-robotics/reachy-mini-emotions-library`](https://huggingface.co/datasets/pollen-robotics/reachy-mini-emotions-library)
- Dances: [`pollen-robotics/reachy-mini-dances-library`](https://huggingface.co/datasets/pollen-robotics/reachy-mini-dances-library)

<details>
<summary>Custom profiles</summary>

Create custom profiles with dedicated instructions and per-profile tool access.

Select and save a startup profile in the UI. The choice is stored in `startup_settings.json`. Before one is saved, `REACHY_MINI_CUSTOM_PROFILE=<name>` can select `profiles/<name>/`; otherwise the app uses `default`.

Every profile directory contains one strict schema-version-1 `profile.md`. TOML metadata is enclosed by `+++`; the remaining Markdown body is the realtime assistant prompt:

```markdown
+++
schema_version = 1
voice = "Aiden"
greeting = "Greet me warmly in one sentence, in character, and vary the wording each time."
hidden = false
default_tools = [
  "dance",
  "camera",
  "sweep_look",
]
+++

## Identity

You are a concise, friendly robot guide.
```

`schema_version`, `default_tools`, and a non-empty Markdown body are required. `voice`, `greeting`, and `hidden` are optional. Set `hidden = true` to omit a profile from the UI. An empty `default_tools` list is valid and inherits nothing.

`default_tools` is the authored baseline. Tools → Tool access stores overrides in instance-local `profile_toolsets.json` without changing bundled profiles. Restoring defaults removes the override. Active-profile changes reconnect the conversation; other changes apply when selected.

Profile directories are data-only. Python tool implementations belong in `src/reachy_mini_conversation_app/tools/`, or in `REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY` for external tools. Each enabled tool ID must resolve to a shared tool, an external tool, or a tool from an installed Hugging Face Space.

See [Creating and adding tools](#creating-and-adding-tools) for the local tool interface and a maintained example.

To manage personalities in the UI:

With `--ui`, Home lists the available profiles and the built-in default:

- Tap a card to apply that personality and start talking.
- Tap "Manage tools" on a saved personality to open its tool access directly.
- Tap "Custom" to create a personality with a name, instructions, and optional greeting. It inherits the default tools, which can be changed under "Manage tools". Managed instances store it at `user_personalities/<name>/profile.md`; standalone runs use `external_content/user_personalities/<name>/profile.md`.

Switching a personality reloads its prompt and effective tools through a quick backend reconnect. Editing `profile.md` directly requires re-selecting the profile or restarting the app.

</details>

<details>
<summary>Locked profile mode</summary>

To create a locked variant of the app that cannot switch profiles, edit `src/reachy_mini_conversation_app/config.py` and set the `LOCKED_PROFILE` constant to the desired profile name:
```python
LOCKED_PROFILE: str | None = "mars_rover"  # Lock to this profile
```
When set, the app ignores saved startup settings, `REACHY_MINI_CUSTOM_PROFILE`, and UI selection. The UI marks the profile as locked and disables editing.

</details>

<a id="external-profiles-and-tools"></a>

<details>
<summary>External profiles and tools</summary>

You can extend the app with profiles/tools stored outside the repository defaults.

- Core profiles are under `profiles/`.
- Core tools are under `src/reachy_mini_conversation_app/tools/`.

Recommended layout:

```text
external_content/
├── external_profiles/
│   └── my_profile/
│       └── profile.md
├── external_tools/
│   └── my_custom_tool.py
├── user_personalities/
│   └── my_custom_profile/
│       └── profile.md
├── installed_tool_spaces.json
└── profile_toolsets.json
```

Environment variables:

Set these values in your `.env` when you want env-driven external profile/tool selection:

```env
# Optional fallback/manual profile selector:
REACHY_MINI_CUSTOM_PROFILE=my_profile
REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY=./external_content/external_profiles
REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY=./external_content/external_tools
# Optional convenience mode:
# AUTOLOAD_EXTERNAL_TOOLS=1
```

Loading rules:

- Profiles: each directory requires a schema-version-1 `profile.md` with explicit `default_tools`; there is no cross-profile fallback.
- Default mode: enabled IDs must resolve to a shared, external, or installed Tool Space tool.
- Autoload: `AUTOLOAD_EXTERNAL_TOOLS=1` adds every valid `*.py` module from `REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY`.
- Web UI: Tools → Tool access enables external modules per profile; it does not upload or edit Python.
- Separation: profile directories contain data only; external Python belongs in `REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY`.
- Tool names: every loaded class needs a unique `Tool.name`; duplicates fail fast.

</details>

<a id="installing-hugging-face-space-tools"></a>

<details>
<summary>Installing Hugging Face Space tools</summary>

You can install MCP-compatible Hugging Face Spaces as remote tool sources for this app. Private Spaces work too, as long as `HF_TOKEN` is set (or you have run `hf auth login`) for an account that can access them. To publish a new Space, follow [Creating and adding tools](#hugging-face-space-tools).

Tools → Tool Spaces installs or refreshes a global source. Its tools then appear under Tools → Tool access for per-profile selection. Removing a Space removes its tools from every profile. Active-profile changes reconnect the conversation; other changes apply when selected.

The app accepts Hugging Face Spaces exposing the standard `/gradio_api/mcp/` endpoint, not arbitrary MCP URLs. Installation discovers the Space's tools and assigns namespaced local IDs, so do not guess or hard-code those IDs beforehand.

```bash
# install + enable in active profile
reachy-mini-conversation-app tool-spaces add <owner/space-name>

# enable in a specific profile
reachy-mini-conversation-app tool-spaces add <owner/space-name> --profile NAME

# install without enabling
reachy-mini-conversation-app tool-spaces add <owner/space-name> --install-only

# list installed spaces
reachy-mini-conversation-app tool-spaces list

# remove an installed space
reachy-mini-conversation-app tool-spaces remove owner/space-name
```

Bundled Pollen Spaces use static specs and are enabled by the default profile. Custom Spaces are validated through the Hugging Face Hub; HF tokens are sent only to private Spaces. Tool metadata is cached in:

- `installed_tool_spaces.json` in the managed app instance directory
- `external_content/installed_tool_spaces.json` in terminal mode

Startup and profile switching read this cache without discovery or MCP probing. Network access occurs only during install, refresh, or remote tool calls. Per-profile access is stored in `profile_toolsets.json` beside the manifest, or under `external_content/` in terminal mode.

Recommended tags for discoverability on Hugging Face:

- `reachy-mini-tool`
- `mcp`

Tags are advisory; installation still requires successful MCP validation.

> [!NOTE]
> Preinstalled Pollen Spaces can be removed like any other (`tool-spaces remove pollen-robotics/reachy-mini-weather-tool`). To restore access, reinstall the Space and restore or update the relevant profile under "Tool access".

</details>

<details>
<summary>Multiple robots on the same subnet</summary>

If you run multiple Reachy Mini daemons on the same network, use:

```bash
reachy-mini-conversation-app --robot-name <name>
```

`<name>` must match the daemon's `--robot-name` value so the app connects to the correct robot.

</details>

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow and [`AGENTS.md`](AGENTS.md) for coding-agent standards.

## License

Apache 2.0
