import enum
import re
import textwrap

from dataclasses import dataclass
from enum import Enum
from typing import (
    ClassVar,
    FrozenSet,
    Optional,
)

from ._base import Cell


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
        yield CellLine(CellLineType.script_start, 0, None, frozenset())

        for line_idx, line in enumerate(lines):
            cell_line = CellLine.from_line(line_idx, line, self.cell_marker)
            if cell_line is not None:
                yield cell_line

        yield CellLine(type=CellLineType.end, line=len(lines), name="", tags=set())

    def _find_cells(self, cell_lines, lines):
        return list(self._iter_find_cells(cell_lines, lines))

    def _iter_find_cells(self, cell_lines, lines):
        def _get_lines(start, end):
            source = "\n".join(lines[idx].rstrip() for idx in range(start, end))
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
            if cell_lines[idx].type in {CellLineType.cell, CellLineType.script_start}:
                end_idx = min(
                    (
                        i
                        for i in range(idx + 1, len(cell_lines))
                        if cell_lines[i].type is CellLineType.cell
                    ),
                    default=len(cell_lines) - 1,
                )
                assert cell_lines[end_idx].type in {CellLineType.cell, CellLineType.end}

                # skip empty cells
                if cell_lines[idx].line != cell_lines[end_idx].line:
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
    script_start = enum.auto()
    cell = enum.auto()
    nested_start = enum.auto()
    nested_end = enum.auto()
    end = enum.auto()


@dataclass
class CellLine:
    type: CellLineType
    line: int
    name: Optional[str]
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
