# `csc` - Tools for non standard Python execution

Install with

```bash
pip install csc
```

## Example

Consider a script to define and train a model

```python
#: parameters
...

#: setup
...

#: train
...

#: save
...
```

To run the the script cell by cell, use:


```python
script = csc.Script("experiment.py")
script.run("parameters")
script.run("setup")
script.run("train")
script.run("save")
```

## Splicing scripts

Different scripts can be "spliced" together by specifying multiple scripts. The
first script acts as the base script and defines the available cells. Subsequent
scripts, spliced scripts, can extend the cells of the base script. All scripts
share a single scope and cells are executed after each other. For each cell,
first the cell of the base script is executed and then any cells of the same
name defined in spliced scripts.


```python
# file: parameters.py
#: parameters
batch_size = ...
```

```python
scripts = csc.Script(["experiment.py", "parameters.py"])

# executes first the 'parameters' cell of 'experiment.py', then the 
# 'parameters' cell of 'parameters.py'
scripts.run("parameters")
```

## License

This package is licensed under the MIT License. See `LICENSE` for details.
