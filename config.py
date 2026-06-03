"""
config.py
=========
Shared helpers for loading/saving sources and slot configurations.

File layout
-----------
sources.json          — list of all video/RTSP sources
slots/<source_id>.json — slot coordinates for each source
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple

SOURCES_FILE = os.path.join(os.path.dirname(__file__), "sources.json")
SLOTS_DIR    = os.path.join(os.path.dirname(__file__), "slots")


# ─────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────

@dataclass
class SlotConfig:
    """Slot layout for one source."""
    rect_w: int = 100
    rect_h: int = 33
    threshold: int = 30
    slots: List[Tuple[int, int]] = field(default_factory=list)
    # Homography / perspective correction (Task 3.1)
    homography_matrix: Optional[List[List[float]]] = None   # 3×3 as nested lists
    warped_size: Optional[List[int]] = None                  # [width, height]
    # User-defined obstacle polygons for A* pathfinding
    # Each polygon is a list of [x, y] points in warped-frame coordinates
    obstacles: List[List[List[int]]] = field(default_factory=list)
    # Polygon slots — each entry is exactly 4 [x, y] points defining a
    # quadrilateral slot boundary (for angled CCTV views).
    # Indexed separately from rect slots; combined count = len(slots) + len(poly_slots)
    poly_slots: List[List[List[int]]] = field(default_factory=list)
    # Optional crop region applied before any processing.
    # Stored as [x, y, width, height] in original-frame pixel coordinates.
    # When set, every pipeline crops the frame to this ROI before detection,
    # so all slot coordinates are relative to the cropped frame.
    crop_region: Optional[List[int]] = None   # [x, y, w, h]


@dataclass
class Source:
    """One video/RTSP source."""
    id: str
    name: str
    uri: str                        # file path, RTSP URL, device index, …
    slots_file: str
    entry_point: Tuple[int, int] = (0, 0)


# ─────────────────────────────────────────────
# sources.json helpers
# ─────────────────────────────────────────────

def load_sources(path: str = SOURCES_FILE) -> List[Source]:
    """Load all sources from sources.json. Returns [] if file missing."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    sources = []
    for s in data.get("sources", []):
        sources.append(Source(
            id=s["id"],
            name=s["name"],
            uri=s["uri"],
            slots_file=s["slots_file"],
            entry_point=tuple(s.get("entry_point", [0, 0])),
        ))
    return sources


def save_sources(sources: List[Source], path: str = SOURCES_FILE) -> None:
    """Persist sources list to sources.json."""
    data = {
        "sources": [
            {
                "id": s.id,
                "name": s.name,
                "uri": s.uri,
                "slots_file": s.slots_file,
                "entry_point": list(s.entry_point),
            }
            for s in sources
        ]
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_source_by_id(source_id: str, path: str = SOURCES_FILE) -> Optional[Source]:
    for s in load_sources(path):
        if s.id == source_id:
            return s
    return None


# ─────────────────────────────────────────────
# slots/<id>.json helpers
# ─────────────────────────────────────────────

def _resolve_slots_path(slots_file: str) -> str:
    """Make slots_file path absolute relative to this file's directory."""
    if os.path.isabs(slots_file):
        return slots_file
    return os.path.normpath(
        os.path.join(os.path.dirname(__file__), slots_file)
    )


def load_slots(source: Source) -> SlotConfig:
    """Load slot config for a source. Returns empty SlotConfig if file missing."""
    path = _resolve_slots_path(source.slots_file)
    if not os.path.exists(path):
        return SlotConfig()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return SlotConfig(
        rect_w=data.get("rect_w", 100),
        rect_h=data.get("rect_h", 33),
        threshold=data.get("threshold", 30),
        slots=[tuple(s) for s in data.get("slots", [])],
        homography_matrix=data.get("homography_matrix", None),
        warped_size=data.get("warped_size", None),
        obstacles=data.get("obstacles", []),
        poly_slots=data.get("poly_slots", []),
        crop_region=data.get("crop_region", None),
    )


def save_slots(source: Source, cfg: SlotConfig) -> None:
    """Persist slot config to the source's slots_file."""
    path = _resolve_slots_path(source.slots_file)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "rect_w": cfg.rect_w,
        "rect_h": cfg.rect_h,
        "threshold": cfg.threshold,
        "slots": [list(s) for s in cfg.slots],
    }
    # Write homography data when matrix is present (Req 2.3)
    if cfg.homography_matrix is not None:
        data["homography_matrix"] = cfg.homography_matrix
        data["warped_size"] = cfg.warped_size  # may be None → writes null
    # Write obstacle polygons (always, even if empty list)
    data["obstacles"] = cfg.obstacles
    # Write polygon slots (always, even if empty list)
    data["poly_slots"] = cfg.poly_slots
    # Write crop region when set
    if cfg.crop_region is not None:
        data["crop_region"] = cfg.crop_region
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[config] Saved {len(cfg.slots)} rect + {len(cfg.poly_slots)} poly slots → {path}")


# ─────────────────────────────────────────────
# CLI helper — add / list sources
# ─────────────────────────────────────────────

def _default_slots_file(source_id: str) -> str:
    return f"./slots/{source_id}.json"


def add_source_interactive() -> None:
    """Prompt the user to add a new source entry to sources.json."""
    print("\n── Add new source ──────────────────────────")
    sid   = input("  ID (no spaces, e.g. 'lot_b'):       ").strip()
    name  = input("  Display name:                        ").strip()
    uri   = input("  URI (file path / RTSP URL / 0 …):   ").strip()
    slots_file = _default_slots_file(sid)

    sources = load_sources()
    if any(s.id == sid for s in sources):
        print(f"[config] Source '{sid}' already exists — skipping.")
        return

    sources.append(Source(id=sid, name=name, uri=uri,
                          slots_file=slots_file, entry_point=(0, 0)))
    save_sources(sources)
    print(f"[config] Source '{sid}' added.")
    print(f"         Run: python mark_parking_slots.py --source {sid}")
    print(f"         Then press 'e' to set the entry point interactively.")


def list_sources() -> None:
    sources = load_sources()
    if not sources:
        print("[config] No sources defined. Run: python config.py --add")
        return
    print(f"\n{'ID':<20} {'Name':<30} {'URI':<40} Slots file")
    print("─" * 100)
    for s in sources:
        print(f"{s.id:<20} {s.name:<30} {s.uri:<40} {s.slots_file}")


if __name__ == "__main__":
    import sys
    if "--add" in sys.argv:
        add_source_interactive()
    elif "--list" in sys.argv:
        list_sources()
    else:
        print("Usage:")
        print("  python config.py --list        List all sources")
        print("  python config.py --add         Add a new source interactively")
