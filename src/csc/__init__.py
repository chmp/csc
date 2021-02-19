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

from ._utils import (
    get,
    get_args,
    main,
    Context as _Context,
    _active_context,
    _exec_code,
    _eval_code,
    _find_cell,
    _parse_script,
    _run_cell,
)


__all__ = ["Script", "get", "get_args", "main"]


class Script:
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
    """

    def __init__(
        self,
        path,
        *,
        cell_marker="%%",
        verbose=True,
    ):
        path = pathlib.Path(path)

        self.path = path
        self.verbose = verbose
        self.cell_marker = str(cell_marker)

        self.ns = types.ModuleType(path.stem)
        self.ns.__file__ = str(path)
        self.ctx = _Context()

        self.cell_pattern = re.compile(
            r"^#\s*" + re.escape(self.cell_marker) + r"(.*)$"
        )

        self._repl = {}

    def repl(self, widget=False, autoclear=False):
        """Enter a REPL to control script execution

        :param widget:
            if ``True``, use an IPython widget for the REPL
        :param autoclear:
            if ``True``, clear the repl output before running the current
            command
        """
        from ._repl import repl

        self._repl["autoclear"] = autoclear
        repl(self, widget=widget)

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

            _run_cell(self, parsed_cells, cell)

    def run_all(self):
        """Run all cells in order define in the script"""
        parsed_cells = self._parse_script()
        for idx in range(len(parsed_cells)):
            if self.verbose and idx != 0:
                print(file=sys.stderr)

            _run_cell(self, parsed_cells, idx)

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
        cell = _find_cell(self._parse_script(), cell)
        return cell.source.splitlines()

    def eval(self, expr: str):
        """Execute an expression inside the script namespace.

        The expression can also be passed as a multiline string::

            result = script.eval('''
                a + b
            ''')

        """
        return _eval_code(self, expr)

    def exec(self, code: str):
        """Execute a Python block inside the script namespace.

        The source is dedented to  allow using  ``.eval`` inside nested
        blocks::

            if cond:
                script.exec('''
                    hello = 'world'
                ''')

        """
        _exec_code(self, code)

    def _parse_script(self) -> List[Tuple[str, str]]:
        with self.path.open("rt") as fobj:
            return _parse_script(fobj, self.cell_pattern)
