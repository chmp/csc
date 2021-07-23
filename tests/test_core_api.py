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


def test_repr_works(script):
    repr(script)
    repr(script["Run"])


def test_names(script):
    assert script.names() == [None, "First", "Second", "Third"]


def test_cells(script):
    assert len(script.names()) == len(script.cells())


def test_run(script):
    script.run()
    assert script.ns.a == 3


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
        (slice_gen["Second", "First"], 1),
        # NOTE: str slices are inclusive
        (slice_gen["First":"Second"], 2),
        (slice_gen["First":"Third", "First"], 1),
    ],
)
def test_selection(script, selection, expected):
    script[selection].run()
    assert script.ns.a == expected


example_script = """
a = 0

#%% First
a = 1

#%% Second
a = 2

#%% Third
a = 3
"""
