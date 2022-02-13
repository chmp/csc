# `csc` - Tools for non standard Python execution

Install with

```bash
pip install csc
```

## Patterns

### Load a script as a module

```python
train_script = csc.load("train.py") 
train_script.train_func()
```

### Extracting local variable from functions

```python
def add(x, y):
    z = x + y
    return z

res = csc.call(add, 1, 2)
assert res.__return__ == 3
assert res.x == 1
assert res.y == 2
assert res.z == 3
```

### Execute scripts with code cells

Consider a script to define and train a model

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

To run the parameters cell, then overwrite the parameters, and finally run set, use:


```python
script = csc.Script("external_script.py")
script["Parameters"].run()
script.ns.hidden_units = 64
script["Setup":].run()
```

### Splicing scripts

Same effect as in the previous example:

```python
script = csc.Script("external_script.py")
with csc.splice(script, "Parameters"):
    script.ns.hidden_units = 64
```

### Using cell tags

To only define the model without training or saving the results, a subset of the
script can be selected via tags:

```python
# execute any cell tagged with "setup"
script[lambda: tags & {"setup"}].run()

# can also be combined with splicing to modify the parameters
with csc.splice(script[lambda: tags & {"setup"}], "Parameters"):
    script.ns.hidden_units = 64
```

### Creating temporary modules

In a Jupyter notebook, first register the magic function

```python
import csc
csc.autoconfig()
```

Afterwards a module can be defined via

```python
%%csc.module my_cool_module

def add(x, y):
    return x + y
```

It can be used as any other module. For example:

```python
import my_cool_module

assert my_cool_module.add(1, 2) == 3
```

## License

This package is licensed under the MIT License. See `LICENSE` for details.

The function `csc._utils.capture_frame` is adapted from this [stackoverflow
post][so-post] by `Niklas Rosenstein` licensed under CC-BY-SA 4.0.

[so-post]: https://stackoverflow.com/a/52358426
