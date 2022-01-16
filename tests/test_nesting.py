import pytest

import csc


@pytest.fixture
def script(tmp_path):
    path = tmp_path / "example.py"
    path.write_text(example_script)
    yield csc.Script(path, cell_marker="%%")


example_script = """
#%% Init
a = 0

#%% Outer
for i in range(10):
    #%% <Inner>
    a += 1

    #%% </Inner>

a = a // 2
"""


def test_parser(script):
    cells = script.cells()
    assert len(cells) == 3

    assert cells[0].name == "Init"
    assert cells[0].range == (2, 4)

    assert cells[1].name == "Outer"
    assert cells[1].range == (5, 12)

    assert cells[2].name == "Inner"
    assert cells[2].range == (7, 9)


def test_exec(script):
    script["Init", "Outer"].run()
    assert script.ns.a == 5

    script["Init", "Inner"].run()
    assert script.ns.a == 1
