"""Output helpers: directories, markdown/CSV tables, JSON, figures."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` (and parents) if needed and return it."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _format_cell(value: object, float_format: str) -> str:
    """Render one table cell (floats via ``float_format``)."""
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return "nan"
        return float_format.format(value)
    return str(value)


def frame_to_markdown(
    df: pd.DataFrame, *, float_format: str = "{:.4g}", index: bool = True
) -> str:
    """Render a DataFrame as a GitHub-flavored markdown table."""
    work = df.reset_index() if index else df
    headers = [str(c) for c in work.columns]
    rows = [
        [_format_cell(v, float_format) for v in row]
        for row in work.itertuples(index=False)
    ]
    widths = [
        max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
        for i, h in enumerate(headers)
    ]
    def fmt_row(cells: list[str]) -> str:
        return "| " + " | ".join(
            c.ljust(w) for c, w in zip(cells, widths)
        ) + " |"

    lines = [
        fmt_row(headers),
        "| " + " | ".join("-" * w for w in widths) + " |",
    ]
    lines.extend(fmt_row(r) for r in rows)
    return "\n".join(lines) + "\n"


def write_table(
    df: pd.DataFrame,
    outdir: Path,
    stem: str,
    *,
    float_format: str = "{:.4g}",
    index: bool = True,
) -> None:
    """Write a DataFrame as both ``stem.csv`` and ``stem.md``."""
    df.to_csv(outdir / f"{stem}.csv", index=index)
    (outdir / f"{stem}.md").write_text(
        frame_to_markdown(df, float_format=float_format, index=index)
    )


def write_json(obj: dict, path: Path) -> None:
    """Write a JSON summary with stable formatting."""
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def save_figure(fig: plt.Figure, path: Path, *, dpi: int = 150) -> None:
    """Save and close a figure."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
