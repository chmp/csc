import argparse
import inspect
import pathlib

import pytest

import csc


def test_example(tmpdir):
    path = pathlib.Path(tmpdir) / "example.py"
    path.write_text(example_script.format(marker="%"))

    example = csc.Script(path, cell_marker="%", verbose=False)
    assert example.list() == [None, "foo", "bar", "baz"]

    example.run("foo")
    assert example.ns.foo == 13

    with pytest.raises(NameError):
        example.run("baz")

    example.run("bar")
    assert example.ns.bar == 21

    example.run("baz")
    assert example.ns.baz == 42

    # inject values at start
    example.exec("bar = 23")
    example.run("baz")
    assert example.ns.bar == 23
    assert example.ns.baz == 46

    example.exec(
        """
        bar = 11
    """
    )
    example.run("baz")
    assert example.ns.bar == 11
    assert example.ns.baz == 22

    assert example.eval("2 * bar") == 22
    assert (
        example.eval(
            """
        2 
        * 
        bar
    """
        )
        == 22
    )


def test_different_marker(tmpdir):
    path = pathlib.Path(tmpdir) / "example.py"
    path.write_text(example_script.format(marker="%%"))

    example = csc.Script(path, cell_marker="%%")
    assert example.list() == [None, "foo", "bar", "baz"]


example_script = """# unmarked cell

#{marker} foo
foo = 13

#{marker} bar
bar = 21

#{marker} baz
baz = 2 * bar"""


def test_getsource(tmpdir):
    func = "def foo():\n    return 42\n"

    path = pathlib.Path(tmpdir) / "example.py"
    path.write_text(getsource_script.format(func=func))

    example = csc.Script(path, cell_marker="%", verbose=False)
    example.run("Cell1")

    actual = inspect.getsource(example.ns.foo)
    assert actual == func


getsource_script = """

#% Cell1
a = 13

{func}

#% Cell2
b = a + 2
"""


def test_argparse_examples(tmpdir):
    path = pathlib.Path(tmpdir) / "example.py"
    path.write_text(argparse_script)

    example = csc.Script(path, cell_marker="%", verbose=False)

    example.run("Run")
    assert example.ns.args.arg == "foo"

    example.ctx.args = ["--arg", "bar"]
    example.run("Run")
    assert example.ns.args.arg == "bar"


argparse_script = """
#% Run
import argparse
import csc

_parser = argparse.ArgumentParser()
_parser.add_argument("--arg", default="foo")

args = _parser.parse_args(csc.get_args())
"""
