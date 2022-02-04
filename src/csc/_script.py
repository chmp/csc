import contextlib
import fnmatch
import inspect
import os
import pathlib
import sys

from dataclasses import dataclass
from types import ModuleType
from typing import (
    Any,
    Iterable,
    List,
    Optional,
    Sequence,
    TextIO,
    Tuple,
    Union,
)

from ._base import ScriptBase, Env, Cell, _normalize_selection
from ._parser import Parser

DEFAULT_CELL_MARKER: str = "%%"


class Script(ScriptBase):
    """A script with cells defined by comments

    :param path:
        The path of the script, can be a string or a :class:`pathlib.Path`.
    :param cell_marker:
        The cell marker used. Cells are defined as ``# {CELL_MARKER} {NAME}``,
        with an arbitrary number of spaces allowed.
    :param args:
        If not ``None``, the command line arguments of the script. While a cell
        is executed, ``sys.argv`` is set to ``[script_name, *args]``.
    :param cwd:
        If not ``None``, change the working directory to it during the script
        execution.

    .. warning::

        Execution of scripts is non threadsafe when the execution environment
        is modified via ``args`` or ``cwd`` as it changes the global Python
        interpreter state.

    """

    def __init__(
        self,
        path: Union[pathlib.Path, str],
        cell_marker: str = DEFAULT_CELL_MARKER,
        args: Optional[Sequence[str]] = None,
        cwd: Optional[Union[str, os.PathLike]] = None,
        verbose: bool = True,
        auto_dedent: bool = True,
    ):
        script_file = ScriptFile(path, cell_marker, auto_dedent=auto_dedent)

        if args is not None:
            args = [script_file.path.name, *args]

        if cwd is not None:
            cwd = pathlib.Path(cwd)

        env = Env(args=args, cwd=cwd)

        self.script_file = script_file
        self.env = env
        self.verbose = verbose

        self.ns = ModuleType(script_file.path.stem)
        self.ns.__file__ = str(script_file.path)
        self.ns.__csc__ = True  # type: ignore

    @property
    def path(self):
        return self.script_file.path

    @property
    def nested(self):
        return NestedCells(self)

    @property
    def cell_marker(self):
        return self.script_file.cell_marker

    def __getitem__(self, selection):
        return ScriptSubset(self, selection)

    def cells(self) -> List["Cell"]:
        return self._cells()

    def _cells(self, nested=False):
        return [cell for cell in self.script_file.parse() if cell.nested == nested]

    def _ipython_key_completions_(self):
        return self.names()

    def _repr_parts_(self) -> Iterable[Tuple[Optional[str], Any]]:
        yield from super()._repr_parts_()
        yield "nested", self.nested.names()


class NestedCells(ScriptBase):
    def __init__(self, script):
        self.script = script

    @property
    def ns(self):
        return self.script.ns

    @property
    def env(self):
        return self.script.env

    @property
    def verbose(self):
        return self.script.verbose

    def cells(self) -> List["Cell"]:
        return self.script._cells(nested=True)

    def __getitem__(self, selection):
        return ScriptSubset(self, selection)


class ScriptSubset(ScriptBase):
    def __init__(self, script, selection):
        self.script = script
        self.selection = selection

    @property
    def ns(self):
        return self.script.ns

    @property
    def env(self):
        return self.script.env

    @property
    def verbose(self):
        return self.script.verbose

    def cells(self) -> List["Cell"]:
        cells = self.script.cells()
        return [cells[idx] for idx in _normalize_selection(cells, self.selection)]

    def __getitem__(self, selection):
        return ScriptSubset(self, selection)


class ScriptFile:
    path: pathlib.Path
    cell_marker: str

    def __init__(
        self, path: Union[pathlib.Path, str], cell_marker: str, auto_dedent=True
    ):
        self.path = pathlib.Path(path).resolve()
        self.cell_marker = cell_marker
        self.auto_dedent = auto_dedent

    def parse(self) -> List["Cell"]:
        with self.path.open("rt") as fobj:
            return self._parse(fobj)

    def _parse(self, fobj: TextIO) -> List["Cell"]:
        return Parser(
            cell_marker=self.cell_marker, auto_dendent=self.auto_dedent
        ).parse(fobj)
