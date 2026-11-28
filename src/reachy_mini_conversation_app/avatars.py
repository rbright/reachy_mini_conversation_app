"""Avatar (SVG) resolution: a profile-local ``avatar.svg``, then the built-in map, then the default."""

import logging
from pathlib import Path

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.profile_store import DEFAULT_PROFILE_NAME, canonical_profile_name


DEFAULT_AVATAR_FILE = "default.svg"
_AVATARS_DIRECTORY = Path(__file__).parent / "static" / "avatars"
logger = logging.getLogger(__name__)

# Built-in profile dir name -> avatar file under static/avatars/.
# Mirror of AVATAR_BY_PROFILE in static/js/constants.js; kept in sync by hand.
AVATAR_BY_PROFILE: dict[str, str] = {
    "bored_teenager": "bored-teenager.svg",
    "captain_circuit": "captain-circuit.svg",
    "chess_coach": "chess-coach.svg",
    "cosmic_kitchen": "cosmic-kitchen.svg",
    "default": "default.svg",
    "hype_bot": "hype-bot.svg",
    "mad_scientist_assistant": "mad-scientist.svg",
    "mars_rover": "mars-rover.svg",
    "nature_documentarian": "nature-doc.svg",
    "noir_detective": "noir-detective.svg",
    "sorry_bro": "sorry-bro.svg",
    "time_traveler": "time-traveler.svg",
    "victorian_butler": "victorian-butler.svg",
}


def _own_avatar_path(name: str) -> Path | None:
    profile_name = canonical_profile_name(name)
    if profile_name == DEFAULT_PROFILE_NAME:
        return None
    try:
        candidate = config.resolve_profile_dir(profile_name) / "avatar.svg"
        return candidate if candidate.is_file() else None
    except OSError as exc:
        logger.warning("Failed to inspect avatar for profile %r: %s", profile_name, exc)
        return None


def avatar_id_for(name: str) -> str:
    """Return a client cache id: the selection for a profile-local avatar, else the shared file's stem."""
    if _own_avatar_path(name) is not None:
        return name
    return AVATAR_BY_PROFILE.get(canonical_profile_name(name), DEFAULT_AVATAR_FILE).removesuffix(".svg")


def read_avatar_svg(name: str) -> str | None:
    """Return a selection's SVG markup, falling back to the default; None if even that is unreadable."""
    own = _own_avatar_path(name)
    if own is not None:
        try:
            return own.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            logger.warning("Failed to read avatar %s: %s", own, exc)

    file = AVATAR_BY_PROFILE.get(canonical_profile_name(name), DEFAULT_AVATAR_FILE)
    for candidate in (_AVATARS_DIRECTORY / file, _AVATARS_DIRECTORY / DEFAULT_AVATAR_FILE):
        if candidate.is_file():
            try:
                return candidate.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                logger.warning("Failed to read avatar %s: %s", candidate, exc)
    return None
