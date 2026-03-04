from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nanobot.config.schema import RoutingConfig


def classify(routing: RoutingConfig, message: str) -> str | None:
    if not routing.rules:
        return None

    lower = message.lower()
    length = len(message)
    valid_models = set(routing.models.keys())

    for rule in routing.rules:
        if rule.use not in valid_models:
            continue
        when = rule.when
        if when.min_length is not None and length < when.min_length:
            continue
        if when.max_length is not None and length > when.max_length:
            continue
        keyword_hit = any(kw.lower() in lower for kw in when.keywords)
        pattern_hit = any(pat in message for pat in when.patterns)
        if keyword_hit or pattern_hit:
            return rule.use

    return None
