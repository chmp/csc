import pathlib

import pytest

import csc


class SliceGen:
    def __getitem__(self, res):
        return res


slice_gen = SliceGen()


@pytest.fixture
def script(tmpdir):
    path = pathlib.Path(tmpdir) / "example.py"
    path.write_text(example_script)
    yield csc.Script(path, cell_marker="%%")


example_script = """
a = 0

#%% First
a = 1

#%% [tag1,tag2] Second
a = 2

#%% [tag1] Third
a = 3
"""


def test_repr_works(script):
    repr(script)
    repr(script["Third"])

    assert "First" in repr(script)
    assert "Third" in repr(script["Third"])
    assert "First" not in repr(script["Third"])

    assert "cells:" in repr(script)
    assert "tags:" in repr(script)
    assert "nested:" in repr(script)

    assert "cells:" in repr(script["Third"])
    assert "tags:" in repr(script["Third"])
    assert "nested:" not in repr(script["Third"])

    assert "cells:" in repr(script.nested)
    assert "tags:" in repr(script.nested)
    assert "nested:" not in repr(script.nested)


def test_names(script):
    assert script.names() == [None, "First", "Second", "Third"]


def test_cells(script):
    assert len(script.names()) == len(script.cells())


def test_run(script):
    script.run()
    assert script.ns.a == 3


def test_selection_with_callable(script):
    assert script[lambda tags: "tag1" in tags].names() == ["Second", "Third"]
    assert script[lambda tags: "tag2" in tags].names() == ["Second"]


def test_selection_with_callable_without_params(script):
    """Lambdas can also be defined without parameters"""
    assert script[lambda: "tag1" in tags].names() == ["Second", "Third"]
    assert script[lambda: "tag2" in tags].names() == ["Second"]


@pytest.mark.parametrize(
    "selection, expected",
    [
        (slice_gen["First"], 1),
        (slice_gen["Second"], 2),
        (
            slice_gen[
                "First",
            ],
            1,
        ),
        (slice_gen["Second", "First"], 1),
        (slice_gen[["Second", "First"]], 1),
        # NOTE: str slices are inclusive
        (slice_gen["First":"Third"], 2),
        (slice_gen["First":, "First"], 1),
    ],
)
def test_selection(script, selection, expected):
    script[selection].run()
    assert script.ns.a == expected


@pytest.mark.parametrize(
    "selection, expected",
    [
        (["First"], 1),
        ([lambda: name == "First"], 1),
        (["Second"], 2),
        (["Second", "First"], 1),
    ],
)
def test_selection_get(script, selection, expected):
    script.get(*selection).run()
    assert script.ns.a == expected


@pytest.mark.parametrize(
    "selection, expected",
    [
        (["First"], 1),
        ([lambda: name == "First"], 1),
        (["Second"], 2),
        (["Second", "First"], 1),
    ],
)
def test_selection_load(script, selection, expected):
    ns = script.get(*selection).load()
    assert ns.a == expected


def test_load_partial(script):
    ns = script["Second"].load()
    assert ns.a == 2


def test_load(tmp_path):
    script_path = tmp_path / "example.py"
    script_path.write_text(example_script)

    ns = csc.load(script_path, select=["First"])
    assert ns.a == 1

    ns = csc.load(script_path, select=["First", "Second"])
    assert ns.a == 2

    ns = csc.load(script_path, select=["Second"])
    assert ns.a == 2

    ns = csc.load(script_path, select=["Second", "First"])
    assert ns.a == 1
