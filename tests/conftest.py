"""Pytest configuration for path setup."""

import os
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).parents[1].resolve()
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


# Make tests reproducible by ignoring machine-specific profile/tool env config.
# Without this, importing config during test collection can pick up a developer's
# local .env and fail before tests run.
os.environ["REACHY_MINI_SKIP_DOTENV"] = "1"
os.environ.pop("REACHY_MINI_CUSTOM_PROFILE", None)
os.environ.pop("REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY", None)
os.environ.pop("REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY", None)


FIXTURE_SCHEMA = """# Schema

```yaml vault-schema
version: 1
vault: Fixture
keys:
  type: text
  created: date
  updated: date
  tags: tags
  author: text
  run: text
  status: text
  state: text
  owner: text
  date: date
  week_of: date
  confidence: text
types:
  emma-session: {class: log, folders: [Emma/Sessions], name: "{date}-{slug}", required: [date]}
  emma-memory: {class: log, folders: [Emma/Weekly Memories], name: "{date}", required: [date, week_of]}
  emma-playbook: {class: log, folders: [Emma/Conversation Playbook], name: "Conversation Playbook - {date}"}
  research: {class: artifact, folders: [Emma/Research], name: "{slug}", values: {confidence: [low, high]}}
agents:
  emma:
    read: [".", Emma/Conversation Playbook, Emma/Sessions, Emma/Weekly Memories, Emma/Research, Private]
    write: [Emma/Sessions, Emma/Weekly Memories, Emma/Research, Private]
```
"""

FIXTURE_AGENTS = """# AGENTS.md

## Mira

Mira plans the week.

## Emma

Emma writes one session log per conversation.

### Tone

Warm and concrete.

## Note rules

Never overwrite a dated note.
"""


@pytest.fixture
def fixture_vault(tmp_path: Path) -> Path:
    """Return a small vault that follows the vault contract."""
    vault = tmp_path / "vault"
    (vault / "System").mkdir(parents=True)
    (vault / "System" / "Schema.md").write_text(FIXTURE_SCHEMA, encoding="utf-8")
    (vault / "AGENTS.md").write_text(FIXTURE_AGENTS, encoding="utf-8")
    playbook = vault / "Emma" / "Conversation Playbook"
    playbook.mkdir(parents=True)
    (playbook / "Current.md").write_text(
        "---\ntype: emma-playbook\ncreated: 2026-09-19\n---\nAsk about the dinosaur book.\n", encoding="utf-8"
    )
    research = vault / "Emma" / "Research"
    research.mkdir(parents=True)
    (research / "approved-topic.md").write_text(
        "---\ntype: research\ncreated: 2026-09-01\nstatus: approved\nauthor: human/ryan\nrun: manual\n---\nDone.\n",
        encoding="utf-8",
    )
    (research / "draft-topic.md").write_text(
        "---\ntype: research\ncreated: 2026-09-02\nstatus: draft\nauthor: agent/emma\nrun: reachy:emma:s0\n---\nWIP.\n",
        encoding="utf-8",
    )
    return vault
