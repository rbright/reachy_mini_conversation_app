"""Reconnect behavior of the settings UI's /rpc client (static/js/api.js), run under Node."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest


HARNESS = Path(__file__).with_name("api_js_harness.mjs")
STATIC_JS = Path(__file__).parents[2] / "src" / "reachy_mini_conversation_app" / "static" / "js"


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
@pytest.mark.parametrize(
    ("scenario", "expected"),
    [
        ("startup", {"attempts": 2, "events": 1}),
        ("unsubscribe", {"attempts": 0}),
        ("single-chain", {"attempts": 8, "warnings": 1}),
        ("outage-recovery", {"attempts": 6, "events": 1, "warnings": 2}),
    ],
)
def test_rpc_reconnect(scenario: str, expected: dict[str, int], tmp_path: Path) -> None:
    """The /rpc client retries for subscribers only, in a single chain, and warns once per outage."""
    js_dir = tmp_path / "js"
    shutil.copytree(STATIC_JS, js_dir)
    # Node before 20.19 / 22.7 loads .js as CommonJS unless package.json says otherwise.
    (js_dir / "package.json").write_text('{"type": "module"}', encoding="utf-8")
    completed = subprocess.run(
        ["node", str(HARNESS), (js_dir / "api.js").as_uri(), scenario],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert {key: result[key] for key in expected} == expected
