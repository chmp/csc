# csc - execute scripts with cells

Install with

```bash
pip install csc
```

Execution of scripts section by section.

Sometimes it may be helpful to run individual parts of a script inside an
interactive environment, for example Jupyter Notebooks. `CellScript` is
designed to support this use case. The basis are Pythn scripts with special cell
annotations. For example consider a script to define and train a model:

```python
#%% Setup
...

#%% Train
...

#%% Save  
...
```


Where each of the `...` stands for arbitrary user defined code. Using
`CellScript` this script can be step by step as:

```python
from csc import CellScript

script = CellScript("external_script.py")

script.run("Setup")
script.run("Train")
script.run("Save")
``` 

To list all available cells use `script.list()`. 
