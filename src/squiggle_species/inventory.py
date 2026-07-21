from __future__ import annotations

from collections import defaultdict
from pathlib import Path


def inventory_ccf5(root: str | Path) -> dict:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in {".ccf5", ".blow5", ".slow5"})
    by_group: dict[str, dict[str, int]] = defaultdict(lambda: {"files": 0, "bytes": 0})
    for path in files:
        relative = path.relative_to(root)
        group = relative.parts[0] if len(relative.parts) > 1 else "root"
        by_group[group]["files"] += 1
        by_group[group]["bytes"] += path.stat().st_size
    return {
        "status": "complete",
        "root": str(root.resolve()),
        "total_files": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "groups": dict(sorted(by_group.items())),
        "extensions": sorted({path.suffix.lower() for path in files}),
    }
