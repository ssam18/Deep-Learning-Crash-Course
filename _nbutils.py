"""Build a .ipynb file from a list of (cell_type, source) tuples.

Used by the build_chapter_<n>.py scripts. Keeps notebook content as plain
Python lists so the build is deterministic and reviewable in plain text.
"""
import json
from pathlib import Path


def md(text):
    return ("markdown", text)


def code(src):
    return ("code", src)


def build_notebook(cells, out_path, kernel_name="python3", display_name="Python 3"):
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
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1)
    return out_path
