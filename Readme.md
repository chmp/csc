# csc - execute scripts one cell at a time

Install with

```bash
pip install csc
```

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
