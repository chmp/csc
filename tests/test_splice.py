import pathlib

import pytest

import csc


@pytest.fixture
def script(tmp_path):
    path = tmp_path / "example.py"
    path.write_text(example_script)
    yield csc.Script(path, cell_marker="%%")


example_script = """
#%% First
a = 1

#%% Second
a += 1
"""


def test_unspliced(script):
    script.run()
    assert script.ns.a == 2


def test_spliced(script):
    with csc.splice(script, lambda: name == "First"):
        script.ns.a = 41

    assert script.ns.a == 42


def test_spliced_exclusive(script):
    with csc.splice(script, lambda: name == "First", inclusive=False):
        script.ns.a = 41

    assert script.ns.a == 2
