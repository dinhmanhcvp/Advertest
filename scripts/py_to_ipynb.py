"""Convert a `# %%`-celled script into an .ipynb (no external deps). Usage: py_to_ipynb.py in.py out.ipynb"""
import json
import sys


def convert(src: str) -> dict:
    cells, cur, kind = [], [], "code"

    def flush():
        nonlocal cur
        body = "\n".join(cur).strip("\n")
        if body:
            if kind == "markdown":
                body = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.splitlines())
            cell = {"cell_type": kind, "metadata": {}, "source": body.splitlines(keepends=True)}
            if kind == "code":
                cell.update(execution_count=None, outputs=[])
            cells.append(cell)
        cur = []

    for line in src.splitlines():
        if line.startswith("# %%"):
            flush()
            kind = "markdown" if "[markdown]" in line else "code"
        else:
            cur.append(line)
    flush()
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"accelerator": "GPU", "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "cells": cells,
    }


if __name__ == "__main__":
    nb = convert(open(sys.argv[1], encoding="utf8").read())
    json.dump(nb, open(sys.argv[2], "w", encoding="utf8"), indent=1)
    print(f"{len(nb['cells'])} cells -> {sys.argv[2]}")
