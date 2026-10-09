"""Convert the jupytext source (nyc_housing_crashes.py) into the executed deliverable notebook.

    python notebook/build_notebook.py
"""
from pathlib import Path

import jupytext
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

HERE = Path(__file__).parent
notebook = jupytext.read(HERE / "nyc_housing_crashes.py")
notebook.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
ExecutePreprocessor(timeout=3600, kernel_name="python3").preprocess(notebook, {"metadata": {"path": str(HERE)}})
nbformat.write(notebook, HERE / "NYC_Housing_x_Crashes.ipynb")
print("written", HERE / "NYC_Housing_x_Crashes.ipynb")
