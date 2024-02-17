"""Execute Python scripts cell by cell
"""
from ._script import Script, VSCODE_CELL_MARKER, DEFAULT_CELL_MARKER

__version__ = "24.2.0"
__all__ = [
    "Script",
    VSCODE_CELL_MARKER,
    DEFAULT_CELL_MARKER,
]
