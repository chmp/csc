# csc - execute scripts one cell at a time

Install with

```bash
pip install csc
```

Sometimes it may be helpful to run individual parts of a script inside an
interactive environment, for example Jupyter Notebooks. `CellScript` is designed
to support this use case. The basis are Python scripts with special cell
annotations. For example consider a script to define and train a model:

```python
#%% Setup
...

#%% Train
...

#%% Save  
...
```


Here each of the `...` stands for arbitrary user defined code. Using
`CellScript` this script can be executed step by step as:

```python
from csc import CellScript

script = CellScript("external_script.py")

# List available cells
script.list()

# Execute cells
script.run("Setup")
script.run("Train")
script.run("Save")

# Open a REPL
script.repl()

# Access variables defined in the script namespace
script.ns.variable
``` 
