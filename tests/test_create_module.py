import pytest

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
