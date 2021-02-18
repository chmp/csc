"""Execution of scripts section by section.

Sometimes it may be helpful to run individual parts of a script inside an
interactive environment, for example Jupyter Notebooks. ``CellScript`` is
designed to support this use case. The basis are Pythn scripts with special cell
annotations. For example consider a script to define and train a model::

    #%% Setup
    ...

    #%% Train
    ...

    #%% Save  
    ...

Where each of the ``...`` stands for arbitrary user defined code. Using
``CellScript`` this script can be step by step as::

    script = CellScript("external_script.py")

    script.run("Setup")
    script.run("Train")
    script.run("Save")

To list all available cells use ``script.list()``. 

The variables defined inside the script can be accessed and modified using the
``ns`` attribute of the script. One example would be to define a parameter cell
with default parameters and the overwrite the values before executing the
remaining cells. Assume the script defines a parameter cell as follows::

    #%% Parameters
    hidden_units = 128
    activation = 'relu'

Then the parameters can be modified as in::

    script.run("Parameters")
    script.ns.hidden_units = 64
    script.ns.activation = 'sigmoid'

"""
import contextlib
import enum
import pathlib
import re
import sys
import textwrap
import types

from typing import Any, Dict, List, Tuple, Union, Optional

from ._parser import _parse_script


__all__ = ["CellScript"]


class CellScript:
    """Allow to execute a python script step by step

    ``CellScript`` is designed to be used inside Jupyter notebooks and allows to
    execute an external script with special cell annotations cell by cell. The
    script is reloaded before execution, but the namespace is persisted on this
    ``CellScript`` instance.

    The namespace of the script is available via the ``ns`` attribute::

        train_script("Setup")
        print("parameters:", sorted(train_script.ns.model.state_dict()))

        train_script("Train")
        train_script.ns.model

    :param path:
        The path of the script, can be a string or a :class:`pathlib.Path`.
    :param cell_marker:
        The cell marker used. Cells are defined as ``# {CELL_MARKER} {NAME}``,
        with an arbitrary number of spaces allowed.
    :param verbose:
        If True, print a summary of the code executed for each cell.
    :param ns:
        The namespace to use for the execution. Per default a new module will
        be constructed. To share the same namespace with the currently running
        notebook it can be set to the ``__main__`` module.
    """

    path: pathlib.Path
    verbose: bool
    cell_marker: str
    ns: Any
    cell_pattern: re.Pattern

    def __init__(
        self,
        path,
        *,
        cell_marker="%%",
        verbose=True,
        ns=None,
    ):
        self.path = pathlib.Path(path)
        self.verbose = verbose
        self.cell_marker = str(cell_marker)
        self.ns = self._valid_ns(ns, self.path)

        self.cell_pattern = re.compile(
            r"^#\s*" + re.escape(self.cell_marker) + r"(.*)$"
        )

    @staticmethod
    def _valid_ns(ns, path):
        if ns is not None:
            return ns

        ns = types.ModuleType(path.stem)
        ns.__file__ = str(path)
        return ns

    def repl(self):
        from ._repl import repl

        repl(self)

    def run(self, *cells: Union[int, str]):
        """Execute cells inside the script

        :param cells:
            The cells to execute. They can be passed as the index of the cell
            or its name. Cell names only have to match the beginning of the
            name, as long as the prefix uniquely defines a cell. For example,
            instead of ``"Setup training"`` also ``"Setup"`` can be used.
        """
        parsed_cells = self._parse_script()
        for idx, cell in enumerate(cells):
            if self.verbose and idx != 0:
                print(file=sys.stderr)

            self._run(parsed_cells, cell)

    def list(self) -> List[Optional[str]]:
        """List the names for all cells inside the script.

        If a cell is unnamed, ``None`` will be returned.
        """
        return [cell.name for cell in self._parse_script()]

    def get(self, cell: Union[int, str]) -> List[str]:
        """Get the source code of a cell

        See the ``run`` method for details of what values can be used for the
        cell parameter.
        """
        cell = self._find_cell(self._parse_script(), cell)
        return cell.source.splitlines()

    def eval(self, expr: str):
        """Execute an expression inside the script namespace.

        The expression can also be passed as a multiline string::

            result = script.eval('''
                a + b
            ''')

        """
        # NOTE add parens to make multiline expressions safe
        return eval("(" + expr.strip() + ")", vars(self.ns), vars(self.ns))

    def exec(self, source: str):
        """Execute a Python block inside the script namespace.

        The source is dedented to  allow using  ``.eval`` inside nested
        blocks::

            if cond:
                script.exec('''
                    hello = 'world'
                ''')

        """
        source = source.strip()
        source = textwrap.dedent(source)

        exec(source, vars(self.ns), vars(self.ns))

    def _parse_script(self) -> List[Tuple[str, str]]:
        with self.path.open("rt") as fobj:
            return _parse_script(fobj, self.cell_pattern)

    def _run(self, parsed_cells: List[Tuple[str, str]], cell: Union[int, str]):
        cell = self._find_cell(parsed_cells, cell)

        if self.verbose:
            self._print_cell(cell.source)

        # include leading new-lines to ensure the line offset of the source
        # matches the file. This is required fo inspect.getsource to work
        # correctly, which in turn is used for example py torch.jit.script
        source = "\n" * cell.range[0] + cell.source

        code = compile(source, str(self.path.resolve()), "exec")

        exec(code, vars(self.ns), vars(self.ns))

    def _find_cell(self, parsed_cells, cell):
        cands = [c for c in parsed_cells if c.matches(cell)]

        if len(cands) == 0:
            raise ValueError("Could not find cell")

        elif len(cands) > 1:
            raise ValueError(
                f"Found multiple cells: {', '.join(str(c.name) for c in cands)}"
            )

        return cands[0]

    def _print_cell(self, cell_source):
        lines = ["> " + line for line in cell_source.strip().splitlines()]

        if len(lines) > 10:
            lines = lines[:9] + ["..."]

        print("\n".join(lines), file=sys.stderr)
