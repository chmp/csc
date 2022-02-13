import pytest

import csc

from csc._utils import _iter_module_parents


@pytest.mark.parametrize(
    "name, parents",
    [
        ("a", []),
        ("a.b", ["a"]),
        ("a.b.c", ["a", "a.b"]),
        ("a.b.c.d", ["a", "a.b", "a.b.c"]),
        ("foo.bar.baz", ["foo", "foo.bar"]),
        ("hello.world", ["hello"]),
    ],
)
def test_iter_module_parents(name, parents):
    assert list(_iter_module_parents(name)) == parents


def test_create_module():
    csc.create_module("csc.test_module_142", module_source)

    from csc.test_module_142 import foo

    assert foo() == 42


module_source = """
def foo():
    return 42
"""
