import re


def parse_script(fobj, cell_marker):
    cell_pattern = re.compile(r"^#\s*" + re.escape(cell_marker) + r"(.*)$")
    return _parse_script(fobj, cell_pattern)


def _parse_script(fobj, cell_pattern):
    cells = []
    current_cell_name = None
    current_cell_lines = []
    current_cell_start = 0

    for idx, line in enumerate(fobj):
        m = cell_pattern.match(line)

        if m is None:
            current_cell_lines.append(line)

        else:
            if current_cell_name is not None or current_cell_lines:
                cell = Cell(
                    current_cell_name,
                    len(cells),
                    (current_cell_start, idx + 1),
                    "".join(current_cell_lines),
                )
                cells.append(cell)

            current_cell_start = idx + 1
            current_cell_name = m.group(1).strip()
            current_cell_lines = []

    # NOTE if current_cell_name is not None or there are lines then idx is defined
    if current_cell_name is not None or current_cell_lines:
        cell = Cell(
            current_cell_name,
            len(cells),
            (current_cell_start, idx + 1),
            "".join(current_cell_lines),
        )
        cells.append(cell)

    return cells


class Cell:
    def __init__(self, name, idx, range, source):
        self.name = name
        self.idx = idx
        self.range = range
        self.source = source

    def matches(self, cell):
        if isinstance(cell, str):
            return self.name is not None and self.name.startswith(cell.strip())

        else:
            return self.idx == cell
