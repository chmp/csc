# `csc` - Tools for non standard execution in Python

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

```python
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

## License

This package is licensed under the MIT License. See `LICENSE` for details.

The function `csc._utils.capture_frame` is adapted from a [stackoverflow
post][so-post] by `Niklas R`, licensed under CC-BY-SA 4.0.

[so-post]: https://stackoverflow.com/a/52358426
