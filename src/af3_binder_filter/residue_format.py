"""Canonical chain-qualified residue serialization with legacy parsing."""

from __future__ import annotations

import re
from typing import Any, Iterable


_QUALIFIED_RESIDUE = re.compile(r"^[^:;,\s]+:(-?\d+)$")
_QUALIFIED_PAIR = re.compile(r"^[^:;,\s]+:(-?\d+)-[^:;,\s]+:(-?\d+)$")
_LEGACY_PAIR = re.compile(r"^(-?\d+):(-?\d+)$")


def format_residue_list(chain_id: str, positions: Iterable[int]) -> str:
    return ";".join(f"{chain_id}:{position}" for position in sorted(set(positions)))


def format_contact_pairs(
    target_chain: str,
    binder_chain: str,
    pairs: Iterable[tuple[int, int]],
) -> str:
    return ";".join(
        f"{target_chain}:{target}-{binder_chain}:{binder}"
        for target, binder in sorted(set(pairs))
    )


def parse_residue_positions(value: Any) -> frozenset[int]:
    """Parse v2 ``A:1;A:2`` and legacy ``1,2`` residue lists."""

    if value is None or value == "":
        return frozenset()
    if isinstance(value, (set, frozenset, list, tuple)):
        return frozenset(int(item) for item in value)
    result: set[int] = set()
    for raw in re.split(r"[;,]", str(value)):
        token = raw.strip()
        if not token:
            continue
        qualified = _QUALIFIED_RESIDUE.fullmatch(token)
        result.add(int(qualified.group(1)) if qualified else int(token))
    return frozenset(result)


def parse_contact_pairs(value: Any) -> frozenset[tuple[int, int]]:
    """Parse v2 chain-qualified and legacy numeric contact-pair lists."""

    if value is None or value == "":
        return frozenset()
    result: set[tuple[int, int]] = set()
    for raw in re.split(r"[;,]", str(value)):
        token = raw.strip()
        if not token:
            continue
        match = _QUALIFIED_PAIR.fullmatch(token) or _LEGACY_PAIR.fullmatch(token)
        if match is None:
            raise ValueError(f"invalid interface residue pair: {token!r}")
        result.add((int(match.group(1)), int(match.group(2))))
    return frozenset(result)


def normalize_residue_list(value: Any, chain_id: str) -> str:
    return format_residue_list(chain_id, parse_residue_positions(value))


def normalize_contact_pairs(
    value: Any,
    target_chain: str,
    binder_chain: str,
) -> str:
    return format_contact_pairs(
        target_chain,
        binder_chain,
        parse_contact_pairs(value),
    )
