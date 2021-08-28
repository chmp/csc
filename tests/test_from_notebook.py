import csc

import json
from pathlib import Path


example_nb = {
    "cells": [
        {
            "cell_type": "code",
            "source": "print('Hello world!')",
            "metadata": {},
        },
        {
            "cell_type": "markdown",
            "source": "Documentation",
            "metadata": {},
        },
    ],
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 4,
}


def test_example(tmp_path: Path):
    nb_path = tmp_path.joinpath("example.ipynb")
    script_path = tmp_path.joinpath("example.py")

    nb_path.write_text(json.dumps(example_nb))

    csc.notebook_to_script(nb_path, script_path)
    script = csc.Script(script_path)

    cells = script.cells()

    assert cells[0].name == "Cell 0"
    assert cells[0].tags == set()

    assert cells[1].name == "Cell 1"
    assert cells[1].tags == {"markdown"}
