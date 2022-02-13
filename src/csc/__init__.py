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
from ._script import Script
from ._utils import (
    autoconfig,
    call,
    create_module,
    export_to_notebook,
    notebook_to_script,
    splice,
    load,
)

__version__ = "22.2.0"
__all__ = [
    "Script",
    "export_to_notebook",
    "notebook_to_script",
    "splice",
    "load",
    "call",
    "autoconfig",
    "create_module",
]
