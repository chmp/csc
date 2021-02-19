"""Utilities and implementation details"""
import contextvars
import contextlib
import re
import sys
import textwrap

csc_context = contextvars.ContextVar("csc_context", default=None)


class Context:
    pass


def get_args():
    """Get the script "arguments"

    * If set, the ``"args"`` key of the script context will be used
    * If the script is executed as a normal python script, the commandline
        args are returned
    * Otherwise, an empty list

    This function can be used to write scripts that can be both executed from
    the commandline and inside a csc Context::

        _parser = argparse.ArgumentParser()
        _parser.add_argument("--arg", default="foo")

        args = _parser.parse_args(csc.get_args())

    When the script is executed as a csc script, the arguments can be
    overwritten via::

        script = csc.Script(...)
        script.ctx.args = ["--arg", "bar"]

    """
    return get(
        "args", lambda in_csc_context: sys.argv[1:] if not in_csc_context else []
    )


def get(key, default_func):
    """Lookup or compute a context variable

    If the script is executed in a csc context and the ``key`` variable is
    set, return it. Otherwise the function ``default_func`` is called with a
    single boolean argument that specifies whether the script is running
    inside a csc context or not.
    """
    ctx = csc_context.get()

    if ctx is not None and hasattr(ctx, key):
        return getattr(ctx, key)

    else:
        return default_func(ctx is not None)


@contextlib.contextmanager
def _active_context(ctx):
    active = csc_context.get()
    if active is not None:
        raise RuntimeError("Cannot nest contexts")

    token = csc_context.set(ctx)
    try:
        yield

    finally:
        csc_context.reset(token)


def parse_script(fobj, cell_marker):
    """Parse a script and return a list of cells"""
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


def _run_cell(script, parsed_cells, cell):
    cell = _find_cell(parsed_cells, cell)

    if script.verbose:
        _print_cell(cell.source)

    # include leading new-lines to ensure the line offset of the source
    # matches the file. This is required fo inspect.getsource to work
    # correctly, which in turn is used for example by torch.jit.script
    source = "\n" * cell.range[0] + cell.source

    code = compile(source, str(script.path.resolve()), "exec")

    with _active_context(script.ctx):
        exec(code, vars(script.ns), vars(script.ns))


def _eval_code(script, code):
    with _active_context(script.ctx):
        # NOTE add parens to make multiline expressions safe
        return eval("(" + code.strip() + ")", vars(script.ns), vars(script.ns))


def _exec_code(script, code):
    code = code.strip()
    code = textwrap.dedent(code)

    with _active_context(script.ctx):
        exec(code, vars(script.ns), vars(script.ns))


def _find_cell(parsed_cells, cell):
    cands = [c for c in parsed_cells if c.matches(cell)]

    if len(cands) == 0:
        raise ValueError("Could not find cell")

    elif len(cands) > 1:
        raise ValueError(
            f"Found multiple cells: {', '.join(str(c.name) for c in cands)}"
        )

    return cands[0]


def _print_cell(cell_source):
    lines = ["> " + line for line in cell_source.strip().splitlines()]

    if len(lines) > 10:
        lines = lines[:9] + ["..."]

    print("\n".join(lines), file=sys.stderr)


def main():
    """A minimalistic cli script to start a script repl"""
    import argparse

    from csc import Script

    _parser = argparse.ArgumentParser()
    _parser.add_argument("--cell-marker", default="%%")
    _parser.add_argument("script")

    args = _parser.parse_args()

    script = Script(args.script, cell_marker=args.cell_marker)
    script.repl()
