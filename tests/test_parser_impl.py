import pytest

from csc._parser import CellLine, CellLineType


@pytest.mark.parametrize(
    "line, expected",
    [
        (
            "#%% hello",
            CellLine(
                type=CellLineType.cell,
                line=42,
                name="hello",
                tags=frozenset(),
            ),
        ),
        (
            "   # %%    hello  ",
            CellLine(
                type=CellLineType.cell,
                line=42,
                name="hello",
                tags=frozenset(),
            ),
        ),
        (
            "#%% [tag1,tag2] hello",
            CellLine(
                type=CellLineType.cell,
                line=42,
                name="hello",
                tags=frozenset({"tag1", "tag2"}),
            ),
        ),
        (
            "#%% [tag1,tag2] <hello>",
            CellLine(
                type=CellLineType.nested_start,
                line=42,
                name="hello",
                tags=frozenset({"tag1", "tag2"}),
            ),
        ),
        (
            "#%% </hello>",
            CellLine(
                type=CellLineType.nested_end,
                line=42,
                name="hello",
                tags=frozenset(),
            ),
        ),
    ],
)
def test_cell_line_examples(line, expected):
    actual = CellLine.from_line(42, line)
    assert actual == expected
