# `csc` - Tools for non standard Python execution

Install with

```bash
pip install csc
```

## Usage

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

### Splicing scripts

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

### CLI

`csc` include a command line script splice and execute scripts. Usage:

```bash
$ python -m csc base_script.py spliced_script_1.py spliced_script_2.py
```

To splice in parameters, `csc` offers a shortcut: arguments specified via `-p
STATEMENT` are concatenated into a script with a single 'parameters' cell that
executes the given statements.

```bash
# this call
$ python -m csc base_script.py -p batch_size=100 -p learning_rate=3e-4

# is equivalent to
$ python -m csc base_script.py parameters.py
$ cat parameters
#: parameters
batch_size=100
learning_rate=3e-4
```

## API Reference

<!-- minidoc "function": "csc.Script", "header_depth": 3 -->
### `csc.Script(base_script: Union[str, pathlib.Path, ForwardRef('FileSource'), ForwardRef('InlineSource')], /, *spliced_scripts, cell_marker: str = '#:', register: bool = False)`

[csc.Script]: #cscscriptbase_script-unionstr-pathlibpath-forwardreffilesource-forwardrefinlinesource-/-spliced_scripts-cell_marker-str--#-register-bool--false

A Python script that can be executed cell by cell

Cells are defined via comments (per default '#: {CELL_NAME}').

#### `csc.Script.list(self) -> List[str]`

[csc.Script.list]: #cscscriptlistself---liststr

List all cells of the script

Only cells of the base script are considered.

#### `csc.Script.run(self, *cell_names) -> None`

[csc.Script.run]: #cscscriptrunself-cell_names---none

Run cells of the script

#### `csc.Script.eval(self, expr: str) -> Any`

[csc.Script.eval]: #cscscriptevalself-expr-str---any

Evaluate an expression the scope of the script

<!-- minidoc -->

<!-- minidoc "function": "csc.FileSource", "header_depth": 3 -->
### `csc.FileSource(path: pathlib.Path, *, cell_marker: str)`

[csc.FileSource]: #cscfilesourcepath-pathlibpath--cell_marker-str

Define a script via a file

<!-- minidoc -->

<!-- minidoc "function": "csc.InlineSource", "header_depth": 3 -->
### `csc.InlineSource(text: str, *, cell_marker: str)`

[csc.InlineSource]: #cscinlinesourcetext-str--cell_marker-str

Define a script via its source

<!-- minidoc -->

## License

This package is licensed under the MIT License. See `LICENSE` for details.
