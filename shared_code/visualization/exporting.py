from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_figure(
    fig,
    directory,
    name,
    formats=("png", "pdf"),
    dpi=300,
):
    """Export a Matplotlib figure in multiple formats."""

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        path = directory / f"{name}.{fmt}"

        fig.savefig(
            path,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
        )


def save_table(
    table,
    directory,
    name,
    formats=("csv", "tex", "html"),
    index=True,
):
    """Export a Pandas table in multiple formats."""

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    if isinstance(table, pd.Series):
        table = table.to_frame()

    for fmt in formats:
        path = directory / f"{name}.{fmt}"

        if fmt == "csv":
            table.to_csv(path, index=index)

        elif fmt == "tex":
            table.to_latex(path, index=index)

        elif fmt == "html":
            table.to_html(path, index=index)