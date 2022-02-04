import contextlib
import os
import pathlib
import sys

from typing import (
    cast,
    Iterator,
    TextIO,
    Union,
)

from ._base import ScriptBase
from ._script import DEFAULT_CELL_MARKER, Script


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


def load(script_path, select=None, cell_marker=DEFAULT_CELL_MARKER):
    """A shortcut for ``Script(script_path)[select].load()``"""
    script = Script(script_path, cell_marker=cell_marker)
    if select is not None:
        if isinstance(select, (tuple, list)):
            script = script.get(*select)

        else:
            script = script.get(select)

    return script.load()


def call(func, /, *args, **kwargs):
    """Call a function an return the locals

    The return value is included as the ``__return__`` attribute.
    """
    captured_frame, result = capture_frame(func, *args, **kwargs)
    return Namespace(
        # NOTE: use dict to allow overriding __return__
        **{
            "__return__": result,
            **captured_frame.f_locals,
        },
    )


def capture_frame(func, /, *args, **kwargs):
    """Call a function and capture the frame

    This function is adapted from https://stackoverflow.com/a/52358426

    Author: Niklas R
    License: CC-BY-SA 4.0
    """
    captured_frame = None
    trace = sys.gettrace()

    def capture_frame(frame, name, arg):
        nonlocal captured_frame, trace

        if captured_frame is None and name == "call":
            captured_frame = frame
            sys.settrace(trace)

        return trace

    sys.settrace(capture_frame)

    try:
        result = func(*args, **kwargs)

    finally:
        sys.settrace(trace)

    return captured_frame, result


class Namespace:
    def __init__(self, /, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


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
