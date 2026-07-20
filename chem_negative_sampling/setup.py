"""Minimal packaging so `python3 -m pc_cng.<module>` works from the research root.

This file only exists to make the pc_cng package importable without manually
setting PYTHONPATH.  It does not declare dependencies (those are handled by the
conda env) and does not modify any existing behaviour.
"""

from setuptools import find_packages, setup

setup(
    name="pc_cng",
    version="0.1.0",
    description="PC-CNG reaction-boundary negative sampling research package",
    packages=find_packages(),
    python_requires=">=3.8",
)
