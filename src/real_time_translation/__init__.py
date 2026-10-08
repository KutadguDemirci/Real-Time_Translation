"""Local-first French-English speech translation application.

The package initializer deliberately stays lightweight: importing the package does not
initialize a GUI, inspect audio devices, or load inference runtimes.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("local-realtime-translation")
except PackageNotFoundError:
    # Source checkouts may be used without installing the project first.
    __version__ = "0.1.0"

__all__ = ["__version__"]

