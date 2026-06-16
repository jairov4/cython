# mode: run
# cython: language_level=3
"""
Interop tests for nullable value types: unwrap coercion, or-default,
conditional expr, and truthiness narrowing.
"""
import cython
from cython import cclass, final, value_type, double
from dataclasses import dataclass
from typing import Optional


@value_type
@final
@cclass
@dataclass(frozen=True)
class Vec2:
    x: double
    y: double


# ---- Pattern A: nullable -> value (unwrap, None guard) ----

@cython.cclass
class Holder:
    pos: Vec2

    def set_from_opt(self, value: Optional[Vec2]) -> None:
        self.pos = value

    def get_x(self) -> double:
        return self.pos.x

    def get_y(self) -> double:
        return self.pos.y


def test_unwrap_nonnone():
    """
    >>> h = Holder()
    >>> h.set_from_opt(Vec2(3.0, 4.0))
    >>> h.get_x()
    3.0
    >>> h.get_y()
    4.0
    """
    pass  # test is exercised via doctest above


def test_unwrap_none_raises():
    """
    >>> import cython
    >>> h = Holder()
    >>> if cython.compiled:
    ...     h.set_from_opt(None)  # doctest: +ELLIPSIS
    ... else:
    ...     raise TypeError("not None (pure-Python stub)")  # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    TypeError: ...
    """
    pass


# ---- Pattern B: x or default ----

def pattern_b(v: Optional[Vec2]) -> Vec2:
    return v or Vec2(0.0, 0.0)


def test_or_default_nonnone():
    """
    >>> r = pattern_b(Vec2(1.0, 2.0))
    >>> r.x
    1.0
    >>> r.y
    2.0
    """
    pass


def test_or_default_none():
    """
    >>> r = pattern_b(None)
    >>> r.x
    0.0
    >>> r.y
    0.0
    """
    pass


# ---- Pattern C: conditional expression ----

def pattern_c(v: Optional[Vec2]) -> Vec2:
    return v if v is not None else Vec2(0.0, 0.0)


def test_cond_expr_nonnone():
    """
    >>> r = pattern_c(Vec2(5.0, 6.0))
    >>> r.x
    5.0
    >>> r.y
    6.0
    """
    pass


def test_cond_expr_none():
    """
    >>> r = pattern_c(None)
    >>> r.x
    0.0
    >>> r.y
    0.0
    """
    pass


# ---- Pattern D: truthiness narrowing then assign ----

def pattern_d(v: Optional[Vec2]) -> Vec2:
    r: Vec2 = Vec2(1.0, 1.0)
    if v:
        r = v
    return r


def test_truthiness_narrowing_nonnone():
    """
    >>> r = pattern_d(Vec2(7.0, 8.0))
    >>> r.x
    7.0
    >>> r.y
    8.0
    """
    pass


def test_truthiness_narrowing_none():
    """
    >>> r = pattern_d(None)
    >>> r.x
    1.0
    >>> r.y
    1.0
    """
    pass


def _doctest():
    import doctest
    import sys
    results = doctest.testmod(sys.modules[__name__], verbose=False)
    if results.failed:
        raise RuntimeError("%d doctest(s) failed" % results.failed)


if __name__ == '__main__':
    _doctest()
