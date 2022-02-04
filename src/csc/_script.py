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

from ._parser import Parser

DEFAULT_CELL_MARKER: str = "%%"


class ScriptBase:
    ns: ModuleType
    env: "Env"
    verbose: bool

    def cells(self) -> List["Cell"]:
        """Return the cells themselves"""
        raise NotImplementedError()

    def run(self) -> None:
        """Run all cells"""
        cells = self.cells()

        if self.verbose:
            print(":: run ", *(repr(cell.name) for cell in cells))

        for cell in cells:
            cell.run(self.ns, self.env)

    def eval(self, expr):
        ns = self.ns
        return eval(expr, vars(ns), vars(ns))

    def dir(self, pattern=None):
        """List all variables inside the scripts namespace

        :param pattern:
            a shell pattern used to filter the variables, e.g.,
            ``script.dir("*_schedule")``
        """
        names = sorted(vars(self.ns))
        if pattern is not None:
            names = [name for name in names if fnmatch.fnmatch(name, pattern)]

        return names

    def source(self):
        """Concatenate the source of all cells"""
        return "\n".join(cell.source for cell in self.cells())

    def names(self) -> List[Union[None, str]]:
        """Return the names of the cells"""
        return [cell.name for cell in self.cells()]

    def split(self, split_point, inclusive=True) -> Tuple["ScriptBase", "ScriptBase"]:
        """Split a script into two parts

        The split point can be any object understood by the item selection. It
        must select a single cell. For ``inclusive=True`` the head  contains
        all cells up to the tail, the second part the rest.  Otherwise, the
        selected cell will be found in the tail.
        """
        split_point = list(_normalize_selection(self.cells(), split_point))
        if len(split_point) != 1:
            raise RuntimeError("split_point must select a single cell")

        split_point = split_point[0]
        split_point = split_point + 1 if inclusive else split_point

        return self[:split_point], self[split_point:]

    def spliced(self, split_point, inclusive=True):
        from ._utils import splice
        return splice(self, split_point, inclusive=inclusive)

    def load(self):
        """Run the current selection and return the namespace"""
        self.run()
        return self.ns

    def get(self, *cells):
        """An alias for __getitem__ to fit into method chains"""
        return self[cells]

    def __getitem__(self, selection):
        raise NotImplementedError()

    def __len__(self):
        return len(self.cells())

    def __iter__(self):
        raise TypeError(
            "Scripts cannot be iterated over. Use .parse() or .names() to iterate "
            "over the cells or their names respectively."
        )

    def __repr__(self) -> str:
        content = " ".join(
            f"{v}" if k is None else f"{k}: {v}" for k, v in self._repr_parts_()
        )
        return "<" + content + ">"

    def _repr_parts_(self) -> Iterable[Tuple[Optional[str], Any]]:
        yield None, type(self).__name__

        try:
            names = self.names()
            tags = self.tags

        except Exception as e:
            yield None, f"invalid {e!r}"

        else:
            yield "cells", names
            yield "tags", sorted(tags)

    @property
    def tags(self):
        return {tag for cell in self.cells() for tag in cell.tags}


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


def _normalize_selection(cells, selection):
    name_to_idx = _LazyeNameToIdxMapper(cells)

    for item in _ensure_list(selection):
        if item is None or isinstance(item, (int, str)):
            yield name_to_idx(item)

        elif isinstance(item, slice):
            start = name_to_idx(item.start) if item.start is not None else None
            stop = name_to_idx(item.stop) if item.stop is not None else None

            cell_indices = range(len(cells))
            yield from cell_indices[start : stop : item.step]

        elif callable(item):
            yield from (
                idx
                for idx, cell in enumerate(cells)
                if _eval_cell_predicate(item, idx, cell)
            )

        else:
            raise ValueError(f"Invalid selector {item}")


def _eval_cell_predicate(predicate, idx, cell):
    """Evaluate a cell predicate

    For details on the semantics see the documentation of :class:`Script`.
    """
    scope = dict(cell=cell, name=cell.name, tags=cell.tags, idx=idx)

    signature = inspect.signature(predicate)

    if not signature.parameters:
        return eval(predicate.__code__, scope, scope)

    has_kwargs = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
    )
    if has_kwargs:
        return predicate(**scope)

    accepted_args = {
        name: scope[name]
        for name, p in signature.parameters.items()
        if p.kind
        in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
    }
    return predicate(**accepted_args)


class _LazyeNameToIdxMapper:
    def __init__(self, cells) -> None:
        self.cells = cells
        self._map = None

    def __call__(self, name_or_idx):
        if isinstance(name_or_idx, int):
            return name_or_idx

        if self._map is None:
            self._map = {}

            for idx, cell in enumerate(self.cells):
                if cell.name in self._map:
                    raise RuntimeError(
                        f"Invalid script file: duplicate cell {cell.name}"
                    )

                self._map[cell.name] = idx

        try:
            return self._map[name_or_idx]

        except KeyError:
            raise RuntimeError(f"Could not find cell {name_or_idx!r}")


def _ensure_list(obj):
    return [obj] if not isinstance(obj, (list, tuple)) else list(obj)


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


class Env:
    """ "Customize the environment the script is executed in"""

    def __init__(self, args: Optional[List[str]], cwd: Optional[pathlib.Path]):
        self.args = args
        self.cwd = cwd

    @contextlib.contextmanager
    def patch(self):
        with self._patch_args(), self._patch_cwd():
            yield

    @contextlib.contextmanager
    def _patch_args(self):
        if self.args is None:
            yield
            return

        prev_args = sys.argv
        sys.argv = self.args

        try:
            yield

        finally:
            sys.argv = prev_args

    @contextlib.contextmanager
    def _patch_cwd(self):
        if self.cwd is None:
            yield
            return

        prev_cwd = os.getcwd()
        os.chdir(self.cwd)

        try:
            yield

        finally:
            os.chdir(prev_cwd)
