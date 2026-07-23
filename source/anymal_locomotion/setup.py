"""Installation script for the ANYmal-D locomotion Isaac Lab extension."""

from __future__ import annotations

import os

import toml
from setuptools import find_packages, setup

EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

setup(
    name="anymal_locomotion",
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    author=EXTENSION_TOML_DATA["package"]["author"],
    maintainer=EXTENSION_TOML_DATA["package"]["maintainer"],
    packages=find_packages(),
    install_requires=["PyYAML>=6.0"],
    python_requires=">=3.11",
    include_package_data=True,
    zip_safe=False,
)
