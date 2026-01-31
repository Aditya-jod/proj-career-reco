import re
from typing import Optional

COUNTRY_ALIASES = {
    "usa": (
        "usa",
        "united states",
        "united states of america",
        "us",
        "america",
        "u.s.",
        "u.s.a",
    ),
    "uk": ("uk", "united kingdom", "england", "britain", "great britain"),
    "uae": ("uae", "united arab emirates", "dubai", "abu dhabi"),
    "south korea": ("south korea", "korea", "republic of korea"),
    "korea": ("south korea", "korea", "republic of korea"),
    "china": ("china", "people's republic of china", "prc"),
    "australia": ("australia", "aus"),
    "canada": ("canada",),
    "germany": ("germany", "deutschland"),
    "france": ("france", "fr"),
    "india": ("india", "bharat"),
}


def normalize_country(value: Optional[str]) -> str:
    if not value:
        return ""
    normalized = value.strip().lower()
    for canonical, aliases in COUNTRY_ALIASES.items():
        if normalized in aliases:
            return canonical
    return normalized


def country_matches(preference: Optional[str], candidate: Optional[str]) -> bool:
    pref = normalize_country(preference)
    cand = normalize_country(candidate)
    return bool(pref and cand and pref == cand)


def country_pattern(value: str) -> str:
    normalized = normalize_country(value)
    aliases = COUNTRY_ALIASES.get(normalized)
    if aliases:
        return "|".join(re.escape(alias) for alias in aliases)
    return re.escape(value)
