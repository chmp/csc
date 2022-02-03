"""Tests that full scrip execution works"""
import pytest

from csc import Script


example_1 = """
def test():
    return 21

a = 42
"""

example_2 = """
def test():
    return 21

#%% Next
a = 42
"""


@pytest.mark.parametrize("source", [example_1, example_2])
def test_run(tmp_path, source):
    script_path = tmp_path / "script.py"
    script_path.write_text(source)

    script = Script(script_path)
    script.run()

    assert script.ns.test() == 21
    assert script.ns.a == 42


@pytest.mark.parametrize("source", [example_1, example_2])
def test_load(tmp_path, source):
    script_path = tmp_path / "script.py"
    script_path.write_text(source)

    ns = Script(script_path).load()
    assert ns.test() == 21
    assert ns.a == 42
