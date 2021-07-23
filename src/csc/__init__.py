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

Where each of the ``...`` stands for arbitrary user defined code. Using
``csc.Script`` this script can be executed step by step as::

    script = csc.Script("external_script.py")

    script["Setup"].run()
    script["Train].run()
    script["Save"].run()

To list all available cells use ``script.names()``.

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

"""
import io
import pathlib
import re
from types import ModuleType

from typing import List, Optional, Tuple, Type, Union


class ScriptBase:
    def run(self):
        for cell in self.cells():
            cell.run(self.ns)

    def names(self):
        return [cell.name for cell in self.cells()]

    def __iter__(self):
        raise TypeError(
            "Scripts cannot be iterated over. Use .parse() or .names() to iterate "
            "over the cells or their names respectively."
        )

    def __repr__(self) -> str:
        self_type = type(self).__name__
        try:
            cells = self.parse()

        except Exception as e:
            return f"<{self_type} invalid {e!r}>"

        cells = [cell.name for cell in cells]
        return f"<{self_type} {cells}>"


class Script(ScriptBase):
    """A script with cells defined by comments

    :param path:
        The path of the script, can be a string or a :class:`pathlib.Path`.
    :param cell_marker:
        The cell marker used. Cells are defined as ``# {CELL_MARKER} {NAME}``,
        with an arbitrary number of spaces allowed.
    """

    def __init__(self, path: Union[pathlib.Path, str], cell_marker: str = "%%"):
        script_file = ScriptFile(path, cell_marker)

        self.script_file = script_file
        self.ns = ModuleType(script_file.path.stem)
        self.ns.__file__ = str(script_file.path)
        self.ns.__csc__ = True

    @property
    def path(self):
        return self.script_file.path

    @property
    def cell_marker(self):
        return self.script_file.cell_marker

    def __getitem__(self, selection):
        return ScriptSubset(self, selection)

    def cells(self) -> List["Cell"]:
        return self.script_file.parse()


class ScriptSubset(ScriptBase):
    def __init__(self, script, selection):
        self.script = script
        self.selection = selection

    @property
    def ns(self):
        return self.script.ns

    def cells(self) -> List["Cell"]:
        cells = self.script.cells()
        return [cells[idx] for idx in _normalize_selection(cells, self.selection)]


def _normalize_selection(cells, selection):
    name_to_idx = _LazyeNameToIdxMapper(cells)

    for item in _ensure_list(selection):
        if item is None or isinstance(item, (int, str)):
            yield name_to_idx(item)

        elif isinstance(item, slice):
            start = name_to_idx(item.start) if item.start is not None else None
            stop = (
                name_to_idx(item.stop) + 1 if isinstance(item.stop, str) else item.stop
            )

            cell_indices = range(len(cells))
            yield from cell_indices[start : stop : item.step]

        else:
            raise ValueError()


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

        return self._map[name_or_idx]


def _ensure_list(obj):
    return [obj] if not isinstance(obj, (list, tuple)) else list(obj)


class ScriptFile:
    path: pathlib.Path
    cell_marker: str
    _cell_pattern: re.Pattern

    def __init__(self, path: Union[pathlib.Path, str], cell_marker: str):
        self.path = pathlib.Path(path).resolve()
        self.cell_marker = cell_marker

        self._cell_pattern = re.compile(r"^#\s*" + re.escape(cell_marker) + r"(.*)$")

    def parse(self) -> List["Cell"]:
        with self.path.open("rt") as fobj:
            return self._parse(fobj)

    def _parse(self, fobj: io.TextIOBase) -> List["Cell"]:
        cells = []
        current_cell_name = None
        current_cell_lines = []
        current_cell_start = 0

        def emit(current_line_idx, next_cell_name):
            nonlocal current_cell_lines, current_cell_name, current_cell_start

            if current_cell_name is not None or any(
                line.strip() for line in current_cell_lines
            ):
                cell = Cell(
                    name=current_cell_name,
                    idx=len(cells),
                    range=(current_cell_start, current_line_idx + 1),
                    source="".join(current_cell_lines),
                )
                cells.append(cell)

            current_cell_start = idx + 1
            current_cell_name = next_cell_name
            current_cell_lines = []

        idx = 0
        for idx, line in enumerate(fobj):
            m = self._cell_pattern.match(line)

            if m is None:
                current_cell_lines.append(line)

            else:
                emit(current_line_idx=idx, next_cell_name=m.group(1).strip())

        emit(current_line_idx=idx, next_cell_name=None)

        return cells


class Cell:
    name: Optional[str]
    idx: int
    range: Tuple[int, int]
    source: str

    def __init__(
        self, name: Optional[str], idx: int, range: Tuple[int, int], source: str
    ):
        self.name = name
        self.idx = idx
        self.range = range
        self.source = source

    def __repr__(self) -> str:
        source = repr(self.source)
        if len(source) > 30:
            source = source[:27] + "..."

        return f"<Cell name={self.name!r} source={source}>"

    def run(self, ns):
        if not hasattr(ns, "__file__"):
            raise RuntimeError("Namespace must have a valid __file__ attribute")

        # include leading new-lines to ensure the line offset of the source
        # matches the file. This is required fo inspect.getsource to work
        # correctly, which in turn is used for example by torch.jit.script
        source = "\n" * self.range[0] + self.source

        code = compile(source, ns.__file__, "exec")
        exec(code, vars(ns), vars(ns))
