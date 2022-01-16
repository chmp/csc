"""Execution of scripts section by section.

Sometimes it may be helpful to run individual parts of a script inside an
interactive environment, for example Jupyter Notebooks. ``csc`` is designed to
support this use case. The basis are Pythn scripts with special cell
annotations. For example consider a script to define and train a model::

    #%% Setup
    ...

    #%% Train
    ...

    #%% Save
    ...

Where each of the ``...`` stands for arbitrary user defined code. Scripts
support selecting individual cells to limit execution to subsets of a script. To
list all available cells use ``script.names()``. In the simplest case, select
one or multiple cells by name::

    script["cell 1"]
    script["cell 1", "cell 2"]

Cells can also be selected by index as in::

    script[0, 1, 2]

Slicing is supported for both names and indices::

    # select all cells up to, but excluding "cell 2"
    script[:"cell 2"]

    # select the first two cells
    script[:2]

For more flexible selections, also callable can be used. The callable can
specify cell properties, such as ``name``, ``idx``, ``tags``, as parameters and
will be called with these properties. A parameter with name cell will be set to
the cell itself::

    script[lambda name: name == "cell 1"]
    script[lambda cell: cell.name == "cell 1"]

Functions without arguments are supported as well::

    script[lambda: name == "cell 1"]

Selections of a script can be executed independently as in::

    script[:"cell 3"].run()
    script["cell 3":].run()


The variables defined inside the script can be accessed and modified using the
``ns`` attribute of the script. One example would be to define a parameter cell
with default parameters and the overwrite the values before executing the
remaining cells. Assume the script defines a parameter cell as follows::

    #%% Parameters
    hidden_units = 128
    activation = 'relu'

Then the parameters can be modified as in::

    script["Parameters"].run()
    script.ns.hidden_units = 64
    script.ns.activation = 'sigmoid'

A common pattern is to execute an initial part of a script, modify the script
namespace, and then continue to evaluate the rest of the script. To simplify
this pattern, scripts support being split::

    head, tail = script.split("Parameters")
    head.run()
    script.ns.parameter = 20
    tail.run()

Or with :func:`slice`::

    with splice(script, "Parameters"):
        script.ns.hidden_units = 64
        script.ns.activation = 'sigmoid'

"""
import contextlib
import enum
import fnmatch
import inspect
import os
import pathlib
import re
import sys
import textwrap

from dataclasses import dataclass
from enum import Enum
from types import ModuleType
from typing import (
    cast,
    ClassVar,
    FrozenSet,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    TextIO,
    Tuple,
    Union,
)

__all__ = ["Script", "export_to_notebook", "notebook_to_script", "splice"]


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
        return splice(self, split_point, inclusive=inclusive)

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
        self_type = type(self).__name__
        try:
            cells = self.cells()

        except Exception as e:
            return f"<{self_type} invalid {e!r}>"

        cell_names = [cell.name for cell in cells]
        cell_tags = sorted({tag for cell in cells for tag in cell.tags})
        return f"<{self_type} cells: {cell_names} tags: {cell_tags}>"

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
        cell_marker: str = "%%",
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


class Cell:
    name: Optional[str]
    range: Tuple[int, int]
    source: str
    tags: Set[str]
    nested: bool

    def __init__(
        self,
        name: Optional[str],
        range: Tuple[int, int],
        source: str,
        tags: Set[str],
        nested: bool,
    ):
        self.name = name
        self.range = range
        self.source = source
        self.tags = tags
        self.nested = nested

    def __repr__(self) -> str:
        source = repr(self.source)
        if len(source) > 30:
            source = source[:27] + "..."

        return f"<Cell name={self.name!r} source={source}>"

    def run(self, ns, env: "Env"):
        if "markdown" in self.tags:
            self._run_markdown(ns, env)

        else:
            self._run_code(ns, env)

    def _run_code(self, ns, env: "Env"):
        if not hasattr(ns, "__file__"):
            raise RuntimeError("Namespace must have a valid __file__ attribute")

        # include leading new-lines to ensure the line offset of the source
        # matches the file. This is required fo inspect.getsource to work
        # correctly, which in turn is used for example by torch.jit.script
        source = "\n" * self.range[0] + self.source

        code = compile(source, ns.__file__, "exec")

        with env.patch():
            exec(code, vars(ns), vars(ns))

    def _run_markdown(self, ns, env: "Env"):
        try:
            from IPython.display import display_markdown

        except ImportError:
            display_markdown = lambda code, raw: print(code)

        source = "\n".join(line[2:] for line in self.source.splitlines())
        display_markdown(source, raw=True)


def export_to_notebook(script, *names):
    """Export the given variables to the ``__main__`` module.

    In a notebook environment the ``__main__`` module is namespace in which
    the user code is executed. By exporting variables to ``__main__``, they
    become global variables inside the notebook scope.
    """
    import __main__

    for name in names:
        setattr(__main__, name, getattr(script.ns, name))


def notebook_to_script(
    notebook: Union[str, os.PathLike, TextIO],
    script: Union[str, os.PathLike, TextIO],
    cell_marker: str = "%%",
):
    """Convert a Jupyter notebook to a script file that csc can parse"""
    import nbformat

    # NOTE: nbformat does not handle Path objects
    with _as_fobj(notebook, mode="r") as notebook_fobj:
        nb = nbformat.read(notebook_fobj, as_version=4)

    unknown_cell_types = set()

    cell_prefixes = {
        "markdown": "[markdown] ",
        "code": "",
    }
    line_prefixes = {
        "markdown": "# ",
        "code": "",
    }

    with _as_fobj(script, mode="w") as script_fobj:
        idx = 0
        for cell in nb.cells:
            if cell.cell_type in {"markdown", "code"}:
                cell_prefix = cell_prefixes[cell.cell_type]
                line_prefix = line_prefixes[cell.cell_type]

                script_fobj.write(f"#{cell_marker} {cell_prefix}Cell {idx}\n")
                for line in cell.source.splitlines():
                    script_fobj.write(f"{line_prefix}{line}\n")

                script_fobj.write("\n")
                idx += 1

            else:
                unknown_cell_types.add(cell.cell_type)

    if unknown_cell_types:
        print(f"Unknown cell types: {unknown_cell_types}", file=sys.stderr)


@contextlib.contextmanager
def splice(script, split_point, inclusive=True) -> Iterator["ScriptBase"]:
    """Split the script, run the first part, yield control and run the second part

    Once use case of this function is to execute initial setup cells, modify
    the script state and then evaluate the rest. For example setup the
    default parameters, modify the paramters, and the run the rest of the
    script.
    """
    head, tail = script.split(split_point, inclusive)

    head.run()

    if script.verbose:
        print(":: run splice")

    yield script

    tail.run()


@contextlib.contextmanager
def _as_fobj(
    path_or_fobj: Union[str, os.PathLike, TextIO],
    mode: str,
) -> Iterator[TextIO]:
    if not isinstance(path_or_fobj, (str, pathlib.Path)):
        yield cast(TextIO, path_or_fobj)

    else:
        with open(path_or_fobj, mode + "t") as fobj:
            yield cast(TextIO, fobj)


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


class Parser:
    def __init__(self, cell_marker="%%", auto_dendent=True):
        self.cell_marker = cell_marker
        self.auto_dendent = auto_dendent

    def parse(self, lines):
        lines = list(lines)
        cell_lines = self._determine_cell_lines(lines)
        cells = list(self._find_cells(cell_lines, lines))
        cells = sorted(cells, key=lambda cell: cell.range[0])

        return cells

    def _determine_cell_lines(self, lines):
        return list(self._iter_determine_cell_lines(lines))

    def _iter_determine_cell_lines(self, lines):
        for line_idx, line in enumerate(lines):
            cell_line = CellLine.from_line(line_idx, line, self.cell_marker)
            if cell_line is not None:
                yield cell_line

        yield CellLine(type=CellLineType.end, line=len(lines), name="", tags=set())

    def _find_cells(self, cell_lines, lines):
        return list(self._iter_find_cells(cell_lines, lines))

    def _iter_find_cells(self, cell_lines, lines):
        def _get_lines(start, end):
            source = "\n".join(lines[idx] for idx in range(start, end))
            return textwrap.dedent(source) if self.auto_dendent else source

        for start_line, end_line in self._iter_cell_ranges(cell_lines):
            yield Cell(
                range=(start_line.line + 1, end_line.line),
                source=_get_lines(start_line.line + 1, end_line.line),
                name=start_line.name,
                tags=start_line.tags,
                nested=start_line.type is CellLineType.nested_start,
            )

    def _iter_cell_ranges(self, cell_lines):
        for idx in range(len(cell_lines)):
            if cell_lines[idx].type is CellLineType.cell:
                end_idx = min(
                    (
                        i
                        for i in range(idx + 1, len(cell_lines))
                        if cell_lines[i].type is CellLineType.cell
                    ),
                    default=len(cell_lines) - 1,
                )
                assert cell_lines[end_idx].type in {CellLineType.cell, CellLineType.end}

                yield cell_lines[idx], cell_lines[end_idx]

            elif cell_lines[idx].type is CellLineType.nested_start:
                # TODO: add proper error messages
                assert (
                    cell_lines[idx + 1].type is CellLineType.nested_end
                    and cell_lines[idx + 1].name == cell_lines[idx].name
                )

                yield cell_lines[idx], cell_lines[idx + 1]

    def _parse_cell_start(self, line):
        m = self._cell_pattern.match(line.strip())
        if m is None:
            return None

        return self._parse_name(m.group("name")), self._parse_tags(m.group("tags"))

    @staticmethod
    def _parse_name(name):
        return name.strip()

    @staticmethod
    def _parse_tags(tags):
        if tags is None:
            return set()

        return {tag.strip() for tag in tags.split(",") if tag.strip()}


class CellLineType(int, Enum):
    cell = enum.auto()
    nested_start = enum.auto()
    nested_end = enum.auto()
    end = enum.auto()


@dataclass
class CellLine:
    type: CellLineType
    line: int
    name: str
    tags: FrozenSet[str]

    _pattern_cache: ClassVar[dict] = {}

    @classmethod
    def from_line(cls, idx, line, cell_marker="%%"):
        pat = cls._compile_pattern(cell_marker)
        m = pat.match(line.strip())
        if m is None:
            return None

        type, name = cls._parse_name(m.group("name"))
        tags = cls._parse_tags(m.group("tags"))

        return CellLine(type=type, line=idx, name=name, tags=tags)

    @classmethod
    def _compile_pattern(cls, cell_marker):
        if cell_marker not in cls._pattern_cache:
            cls._pattern_cache[cell_marker] = re.compile(
                r"^#\s*"
                + re.escape(cell_marker)
                + r"\s+(?:\[(?P<tags>[\w,]+)\])?(?P<name>.*)$"
            )

        return cls._pattern_cache[cell_marker]

    @staticmethod
    def _parse_name(name):
        name = name.strip()

        if name.startswith("</"):
            assert name.endswith(">")
            return CellLineType.nested_end, name[2:-1].strip()

        elif name.startswith("<"):
            assert name.endswith(">")
            return CellLineType.nested_start, name[1:-1].strip()

        else:
            return CellLineType.cell, name

    @staticmethod
    def _parse_tags(tags):
        if tags is None:
            return set()

        return frozenset(tag.strip() for tag in tags.split(",") if tag.strip())
