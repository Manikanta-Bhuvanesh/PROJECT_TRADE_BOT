from __future__ import annotations

import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def file_size(p: Path) -> int:
    return p.stat().st_size if p.is_file() else 0


def prepare_csv_for_telegram(
    csv_path: Path,
    *,
    max_bytes: int,
    work_dir: Path,
) -> list[Path]:
    """
    Return paths to upload: original CSV, a zip, or multiple CSV row-slices.
    """
    sz = file_size(csv_path)
    if sz <= 0:
        return []
    if sz <= max_bytes:
        return [csv_path]

    work_dir.mkdir(parents=True, exist_ok=True)
    zip_path = work_dir / f"{csv_path.stem}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname=csv_path.name)
    if file_size(zip_path) <= max_bytes:
        return [zip_path]

    df = pd.read_csv(csv_path)
    n = len(df)
    if n == 0:
        return [csv_path]

    parts_needed = max(2, int(math.ceil(sz / float(max_bytes))))
    splits = np.array_split(df, parts_needed)
    out: list[Path] = []
    for k, chunk in enumerate(splits, start=1):
        p = work_dir / f"{csv_path.stem}_part{k:02d}.csv"
        chunk.to_csv(p, index=False)
        out.append(p)

    # If a single slice is still huge (wide rows), subdivide that file only.
    final: list[Path] = []
    for p in out:
        if file_size(p) <= max_bytes:
            final.append(p)
            continue
        sub = pd.read_csv(p)
        sub_parts = max(2, int(math.ceil(file_size(p) / float(max_bytes))))
        for j, sc in enumerate(np.array_split(sub, sub_parts), start=1):
            sp = work_dir / f"{p.stem}_sub{j:02d}.csv"
            sc.to_csv(sp, index=False)
            final.append(sp)
        p.unlink(missing_ok=True)
    return final
