import pytest

import csc


def test_call():
    def func(arg):
        a = 42
        return arg

    ns = csc.call(func, 21)

    assert ns.__return__ == 21
    assert ns.a == 42


def test_exceptions():
    def func():
        raise RuntimeError()

    with pytest.raises(RuntimeError):
        csc.call(func)


def test_example():
    def add(x, y):
        z = x + y
        return z

    res = csc.call(add, 1, 2)
    assert res.__return__ == 3
    assert res.x == 1
    assert res.y == 2
    assert res.z == 3
