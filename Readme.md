# csc - execute scripts one cell at a time

## Installation

Install with

```bash
pip install csc
```

## Usage

Sometimes it may be helpful to run individual parts of a script inside an
interactive environment, for example Jupyter Notebooks. ``csc`` is designed to
support this use case. The basis are Pythn scripts with special cell
annotations. For example consider a script to define and train a model:

```python
#%% [setup] Parameters
...

#%% [setup] Setup
...

#%% Train
...

#%% Save
...
```

Where each of the ``...`` stands for arbitrary user defined code. And the names
inside brackets are tags. Using ``csc.Script`` this script can be executed as:

```python
script = csc.Script("external_script.py")
script.run()
```

To modify the parameters, e.g., modify the number of hidden units, the script
can be executed step by step. The variables defined inside the script can be
accessed and modified using the ``ns`` attribute of the script. For example:

```python
script["Parameters"].run()
script.ns.hidden_units = 64
scripts["Setup":].run()
```

To support this common pattern, `csc` offers the splice the function:

```python
with csc.splice(script, "Parameters"):
    scripts.ns.hidden_units = 64
```

To only define the model without training or saving the results, a subset of the
script can be selected via tags:

```python
# execute any cell tagged with "setup"
scripts[lambda: "setup" in tags].run()

# can also be combined with splicing to modify the parameters
with csc.splice(script[lambda: "setup" in tags], "Parameters):
    script.ns.hidden_units = 64

# use the model
model = script.ns.model
```

## License

This package is licensed under the MIT License. See `LICENSE` for details.
