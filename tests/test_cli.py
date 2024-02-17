from pathlib import Path

import pytest

from csc._script import DEFAULT_CELL_MARKER
from csc._cli import parse_args, main, get_assigned_names, Arguments

script_experiment_path = (
    Path(__file__).parent.joinpath("script_experiment.py").resolve(strict=True)
)
script_simple_path = (
    Path(__file__).parent.joinpath("script_simple.py").resolve(strict=True)
)


def test_parser__error__no_scripts():
    with pytest.raises(SystemExit):
        parse_args([])


@pytest.mark.parametrize(
    "args, expected",
    [
        pytest.param(
            ["a.py", "b.py"],
            Arguments(
                cell_marker=DEFAULT_CELL_MARKER,
                register=False,
                files=(Path("a.py"), Path("b.py")),
                parameters=(),
            ),
        ),
        pytest.param(
            ["-p", "a = 42", "-p", "b = 21", "a.py"],
            Arguments(
                cell_marker=DEFAULT_CELL_MARKER,
                register=False,
                files=(Path("a.py"),),
                parameters=("a = 42", "b = 21"),
            ),
            id="parameter overrides",
        ),
    ],
)
def test_parser__scripts(args, expected):
    assert parse_args(args) == expected


def test_experiment__example():
    script = main([f"{script_experiment_path}", "-p", "batch_size = 32"])
    assert script.list() == ["", "parameters", "train"]

    assert script.scope.batch_size == 32
    assert script.scope.model == "<Model learning_rate=0.0003 batch_size=32>"


def test_experiment__overwrite_warning():
    """Test that csc warns for undefined parameters that are overwritten"""
    with pytest.warns(UserWarning, match="Parameter foo_bar is not defined"):
        main([f"{script_experiment_path}", "-p", "foo_bar = None"])


def test_simple__example():
    """Test that csc warns for scripts without parameter block, but with overwrites"""
    with pytest.warns(UserWarning, match=r"CLI called with parameters.*"):
        main([f"{script_simple_path}", "-p", "batch_size = 32"])


@pytest.mark.parametrize(
    "source, names",
    [
        pytest.param("", ()),
        pytest.param("def foo(): pass", ()),
        pytest.param("a = 42", ("a",)),
        pytest.param("a = 21\nb=42", ("a", "b")),
    ],
)
def test_get_assigned_names(source: str, names: tuple[str, ...]):
    assert get_assigned_names(source) == names
