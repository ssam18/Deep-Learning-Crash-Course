"""Build a .ipynb file from a list of (cell_type, source) tuples.

Used by the build_chapter_<n>.py scripts. Keeps notebook content as plain
Python lists so the build is deterministic and reviewable in plain text.

`build_notebook` automatically prepends two cells to every chapter notebook:
1. A markdown cell with Colab + Binder "Open online" badges that link back
   to the chapter's notebook in the public GitHub repo.
2. A code cell (commented out by default) that installs the project
   requirements when the notebook is opened in Colab / Binder. Local users
   can leave it untouched; cloud users uncomment the relevant line.
"""
import json
from pathlib import Path

REPO = "ssam18/Deep-Learning-Crash-Course"
REQUIREMENTS_URL = f"https://raw.githubusercontent.com/{REPO}/main/requirements.txt"


def md(text):
    return ("markdown", text)


def code(src):
    return ("code", src)


def _badge_cell(rel_path):
    colab_url = f"https://colab.research.google.com/github/{REPO}/blob/main/{rel_path}"
    binder_url = f"https://mybinder.org/v2/gh/{REPO}/main?filepath={rel_path}"
    return md(
        f"[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab_url}) "
        f"[![Open in Binder](https://mybinder.org/badge_logo.svg)]({binder_url})\n"
        "\n"
        "> Click a badge above to run this notebook in your browser. Colab uses your Google account; Binder is anonymous. The next cell installs the project dependencies — uncomment one line if you are on Colab or Binder.\n"
    )


def _install_cell():
    return code(
        "# Cloud setup — uncomment ONE line if you are running this notebook in Colab or Binder.\n"
        "# Local users with `pip install -r requirements.txt` already done can leave both commented.\n"
        "\n"
        "# Colab: install everything the chapter needs from the repo's requirements.txt\n"
        f"# !pip install -q -r {REQUIREMENTS_URL}\n"
        "\n"
        "# Binder: dependencies are already provisioned from requirements.txt at container start, no action needed.\n"
    )


def build_notebook(cells, out_path, kernel_name="python3", display_name="Python 3"):
    out_path = Path(out_path)
    # Compute the path relative to the repo root for badge URLs.
    rel_path = out_path.relative_to(Path(__file__).parent).as_posix()
    cells = [_badge_cell(rel_path), _install_cell()] + list(cells)

    nb_cells = []
    for ctype, source in cells:
        if isinstance(source, str):
            lines = source.splitlines(keepends=True)
        else:
            lines = list(source)
        cell = {
            "cell_type": ctype,
            "metadata": {},
            "source": lines,
        }
        if ctype == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        nb_cells.append(cell)
    nb = {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {
                "display_name": display_name,
                "language": "python",
                "name": kernel_name,
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1)
    return out_path
