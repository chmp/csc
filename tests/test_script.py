from pathlib import Path

from csc import Script

script_simple_path = (
    Path(__file__).parent.joinpath("script_simple.py").resolve(strict=True)
)


def test_example__script_simple():
    script = Script(script_simple_path)
    assert script.list() == ["", "first", "second"]

    script.run("first")
    assert script.scope.a == 21
    assert not hasattr(script.scope, "b")

    script.run("second")
    assert script.scope.a == 21
    assert script.scope.b == 42
