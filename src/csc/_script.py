import re
import sys

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Union, Iterable
from textwrap import dedent


DEFAULT_CELL_MARKER = r"#:"
VSCODE_CELL_MARKER = r"#\s*%%"

PathLike = Union[str, Path]
SourceLike = Union[PathLike, "FileSource", "InlineSource"]


class Script:
    def __init__(
        self,
        base_script: SourceLike,
        /,
        *spliced_scripts: SourceLike,
        cell_marker: str = DEFAULT_CELL_MARKER,
        register: bool = False,
    ):
        self._sources = self._ensure_sources(
            [base_script, *spliced_scripts], cell_marker=cell_marker
        )
        self.scope = self._build_scope(self._sources[0])
        self.register = register

    @staticmethod
    def _ensure_sources(
        source: Iterable[Union[PathLike, "FileSource", "InlineSource"]],
        cell_marker: str,
    ) -> list[Union["FileSource", "InlineSource"]]:
        res: list["FileSource" | "InlineSource"] = []
        for item in source:
            if isinstance(item, (str, Path)):
                res.append(FileSource(path=Path(item), cell_marker=cell_marker))

            elif isinstance(item, (InlineSource, FileSource)):
                res.append(item)

            else:
                raise RuntimeError(f"Invalid source {item}")

        if not res:
            raise RuntimeError("Need at least one source")

        return res

    @staticmethod
    def _build_scope(source: Union["FileSource", "InlineSource"]) -> ModuleType:
        name = source.file.stem if source.file is not None else "<unnamed>"
        scope = ModuleType(name)
        scope.__name__ = name
        scope.__file__ = str(source.file) if source.file is not None else None

        return scope

    def eval(self, expr: str) -> Any:
        return eval(expr, self.scope.__dict__, self.scope.__dict__)

    def run(self, *cell_names: str):
        for cell_name in cell_names:
            for cell in self.get(cell_name):
                self._run(cell)

    def get(self, cell_name: str) -> tuple["Cell", ...]:
        res: list["Cell"] = []

        for idx, source in enumerate(self._sources):
            for cell in source.parse():
                if cell.name != cell_name:
                    continue

                res.append(cell)

            if idx == 0 and not res:
                return tuple()

        return tuple(res)

    def _run(self, cell: "Cell"):
        print(cell.source[:100])
        # NOTE: use compile -> exec to allow specifying the source path
        code = compile(
            dedent(cell.source_with_offset),
            cell.file if cell.file is not None else "<unnamed>",
            mode="exec",
        )
        with register_module(self.register, self.scope):
            exec(
                code,
                self.scope.__dict__,
                self.scope.__dict__,
            )

    def list(self):
        return [cell.name for cell in self._sources[0].parse()]


class InlineSource:
    _text: str
    _cell_marker: str
    file: Path | None

    def __init__(self, text: str, *, cell_marker: str):
        self._text = text
        self._cell_marker = cell_marker
        self.file = None

    def parse(self) -> list["Cell"]:
        return parse_script(self._text, cell_marker=self._cell_marker, file=None)


class FileSource:
    _path: Path
    _cell_marker: str
    file: Path | None

    def __init__(self, path: Path, *, cell_marker: str):
        self._path = path
        self._cell_marker = cell_marker
        self.file = path

    def parse(self) -> list["Cell"]:
        return parse_script(
            self._path.read_text(), cell_marker=self._cell_marker, file=self._path
        )


@contextmanager
def register_module(active: bool, module: ModuleType):
    if not active:
        yield

    else:
        if module.__name__ in sys.modules:
            raise ValueError("cannot overwrite existing module")

        sys.modules[module.__name__] = module
        try:
            yield

        finally:
            del sys.modules[module.__name__]


@dataclass
class Cell:
    name: str
    """The name of the cell"""
    file: Path | None
    """The file from which this cell was parsed"""
    offset: int
    """The offset of this cell in the script file"""
    source: str
    """The raw source code of the cell without additional buffers"""

    @property
    def source_with_offset(self):
        """The source with additional empty lines.

        The source is prefixed with enough empty lines that the line numbers are
        the same as those in the full script"""
        return "\n" * self.offset + self.source


def parse_script(
    script_source,
    *,
    cell_marker=DEFAULT_CELL_MARKER,
    file: Path | None = None,
) -> list[Cell]:
    def _finalize_head():
        head_name, head_offset, head_fragments = cell_stack.pop()
        cells.append(
            Cell(
                name=head_name,
                file=file,
                offset=head_offset,
                source="\n".join(head_fragments),
            )
        )

    def _ensure_no_nesting():
        if len(cell_stack) > 1:
            raise ValueError(
                "Unclosed nested cells: "
                f"{[cell_name for cell_name, _, _ in cell_stack[1:]]}"
            )

    def _append_all(cell_source):
        for _, _, fragments in cell_stack:
            fragments.append(cell_source)

    cell_stack: list[tuple[str, int, list[str]]] = []
    cells: list[Cell] = []

    for name, offset, source in split_script(script_source, cell_marker=cell_marker):
        if name.startswith("+"):
            cell_stack.append((name[1:], offset, []))
            _append_all(source)

        elif name.startswith("-"):
            if cell_stack[-1][0] != name[1:]:
                raise ValueError(
                    f"Invalid close tag: expected {cell_stack[-1][0]!r}, "
                    f"found {name!r}"
                )

            _finalize_head()
            _append_all(source)

        else:
            _ensure_no_nesting()
            if cell_stack:
                _finalize_head()

            cell_stack.append((name, offset, [source]))

    _ensure_no_nesting()

    if cell_stack:
        _finalize_head()

    return cells


def split_script(
    source, *, cell_marker=DEFAULT_CELL_MARKER
) -> list[tuple[str, int, str]]:
    pattern = r"^\s*" + cell_marker + r"(.*)$"
    cells: list[tuple[str, int, list[str]]] = [("", 0, [])]

    for idx, line in enumerate(source.splitlines()):
        if (m := re.match(pattern, line)) is not None:
            name = m.group(1).strip()
            cells.append((name, idx, [line]))

        else:
            cells[-1][2].append(line)

    return [(name, offset, "\n".join(lines)) for name, offset, lines in cells]
