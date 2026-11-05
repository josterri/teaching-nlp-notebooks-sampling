#!/usr/bin/env python3
"""Build the notebooks from src/*.py (percent format), execute them, keep the outputs.

    python tools/build_notebooks.py                 build and execute 01 and 02
    python tools/build_notebooks.py 02              one notebook
    python tools/build_notebooks.py --figures DIR   also write every PNG output to DIR

Source format: a line `# %% [markdown]` opens a markdown cell whose lines start with `# `;
`# %%` opens a code cell; `# %% include setup.py` inserts src/setup.py as a code cell.
A code cell starting with `# @title` is folded in Colab and in JupyterLab.
Widget views are dropped from the stored file; an error inside a slider's output fails the build.
"""
import argparse
import base64
import pathlib
import re
import sys
import time

import nbformat
from nbclient import NotebookClient

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
NOTEBOOKS = {"01": "01_sampling_visual", "02": "02_sampling_mathematics"}
WIDGET_VIEW = "application/vnd.jupyter.widget-view+json"
WIDGET_STATE = "application/vnd.jupyter.widget-state+json"


def code_cell(text):
    cell = nbformat.v4.new_code_cell(text)
    if text.startswith("# @title"):
        cell.metadata.update({"cellView": "form", "jupyter": {"source_hidden": True}})
    return cell


def parse(path):
    cells, kind, lines = [], None, []

    def flush():
        text = "\n".join(lines).strip("\n")
        if kind == "markdown":
            text = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in text.splitlines())
            cells.append(nbformat.v4.new_markdown_cell(text))
        elif kind and kind.startswith("include "):
            cells.append(code_cell((SRC / kind.split()[1]).read_text().strip("\n")))
        elif kind == "code":
            cells.append(code_cell(text))

    for line in path.read_text().splitlines():
        m = re.match(r"# %%(.*)$", line)
        if m:
            flush()
            arg = m.group(1).strip()
            kind = "markdown" if arg == "[markdown]" else arg if arg.startswith("include ") else "code"
            lines = []
        else:
            lines.append(line)
    flush()
    return cells


def widget_errors(nb):
    state = nb.metadata.get("widgets", {}).get(WIDGET_STATE, {}).get("state", {})
    found = []
    for model in state.values():
        for out in model.get("state", {}).get("outputs", []):
            if out.get("output_type") == "error":
                found.append(f"{out.get('ename')}: {out.get('evalue')}")
    return found


def build(key, figures):
    name = NOTEBOOKS[key]
    nb = nbformat.v4.new_notebook()
    nb.cells = parse(SRC / f"{name}.py")
    nb.metadata.update({
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python"},
        "colab": {"provenance": [], "toc_visible": True},
    })
    t0 = time.time()
    NotebookClient(nb, timeout=900, kernel_name="python3", record_timing=False, store_widget_state=True,
                   resources={"metadata": {"path": str(ROOT)}}).execute()
    errors = widget_errors(nb)
    if errors:
        sys.exit(f"{name}: error inside a slider output: " + "; ".join(errors))
    nb.metadata.pop("widgets", None)
    n_png = 0
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        cell.outputs = [o for o in cell.outputs if WIDGET_VIEW not in o.get("data", {})]
        for o in cell.outputs:
            png = o.get("data", {}).get("image/png")
            if png and figures:
                n_png += 1
                (figures / f"{key}-cell{i:02d}-{n_png:02d}.png").write_bytes(base64.b64decode(png))
    nbformat.validate(nb)
    nbformat.write(nb, ROOT / f"{name}.ipynb")
    print(f"{name}.ipynb: {len(nb.cells)} cells, executed in {time.time() - t0:.0f} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="*", default=sorted(NOTEBOOKS))
    ap.add_argument("--figures", type=pathlib.Path)
    args = ap.parse_args()
    if args.figures:
        args.figures.mkdir(parents=True, exist_ok=True)
    for key in args.which:
        build(key, args.figures)


if __name__ == "__main__":
    main()
