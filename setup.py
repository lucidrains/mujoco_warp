"""Build hooks for the fork's published artifacts.

The test modules (`*_test.py`) live next to the sources upstream; keep them in the
repository but leave them out of the wheel and sdist (`MANIFEST.in` handles the
sdist, this hook handles the wheel).
"""

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py


class BuildPy(_build_py):
  """build_py that skips test modules."""

  def find_package_modules(self, package, package_dir):
    return [
      (pkg, mod, path)
      for pkg, mod, path in super().find_package_modules(package, package_dir)
      if not mod.endswith("_test") and mod != "conftest"
    ]


setup(cmdclass={"build_py": BuildPy})
