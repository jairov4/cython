# mode: run
# cython: language_level=3
"""
Nullable value type as function parameter, return value, and default argument.

Covers:
 - nullable Optional[Vec2] param in def / @cfunc / @ccall
 - nullable return type from def / @cfunc / @ccall
 - default parameter value = None
 - passing both None and a real value
 - chained calls
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

    def mag2(self) -> double:
        return self.x * self.x + self.y * self.y


# ---------------------------------------------------------------------------
# def functions returning Optional[Vec2] directly (Bug 1 regression test).
# A plain `def` annotated as -> Optional[Vec2] must return PyObject*, not
# the nullable struct.  The compiler should box automatically.

def def_return_some() -> Optional[Vec2]:
    """
    >>> r = def_return_some()
    >>> r is not None
    True
    >>> isinstance(r, Vec2)
    True
    >>> abs(r.x - 1.0) < 1e-9 and abs(r.y - 2.0) < 1e-9
    True
    """
    return Vec2(1.0, 2.0)


def def_return_none() -> Optional[Vec2]:
    """
    >>> def_return_none() is None
    True
    """
    return None


def def_return_conditional(flag: object) -> Optional[Vec2]:
    """
    >>> r = def_return_conditional(True)
    >>> r is not None and isinstance(r, Vec2)
    True
    >>> def_return_conditional(False) is None
    True
    """
    if flag:
        return Vec2(3.0, 4.0)
    return None


# ---------------------------------------------------------------------------
# def functions with nullable params; return scalar or box to object

def sum_if_both(a: Optional[Vec2], b: Optional[Vec2]) -> object:
    """
    >>> r = sum_if_both(Vec2(1.0, 2.0), Vec2(3.0, 4.0))
    >>> r is not None and isinstance(r, Vec2)
    True
    >>> sum_if_both(None, Vec2(1.0, 0.0)) is None
    True
    >>> sum_if_both(Vec2(1.0, 0.0), None) is None
    True
    >>> sum_if_both(None, None) is None
    True
    """
    if a is None or b is None:
        return None
    result: Vec2 = Vec2(a.x + b.x, a.y + b.y)
    return result


def get_x_or_default(o: Optional[Vec2], default: double = -1.0) -> double:
    """
    >>> get_x_or_default(Vec2(3.0, 4.0))
    3.0
    >>> get_x_or_default(None)
    -1.0
    >>> get_x_or_default(None, 99.0)
    99.0
    """
    if o is not None:
        return o.x
    return default


# ---------------------------------------------------------------------------
# Default parameter = None

def first_non_none(a: Optional[Vec2] = None, b: Optional[Vec2] = None) -> object:
    """
    >>> first_non_none() is None
    True
    >>> r = first_non_none(Vec2(1.0, 2.0))
    >>> r is not None and isinstance(r, Vec2)
    True
    >>> r2 = first_non_none(None, Vec2(5.0, 6.0))
    >>> r2 is not None and isinstance(r2, Vec2)
    True
    """
    if a is not None:
        boxed: object = a
        return boxed
    if b is not None:
        boxed2: object = b
        return boxed2
    return None


# ---------------------------------------------------------------------------
# @cython.cfunc with nullable param + return

@cython.cfunc
def _cfunc_scale(o: Optional[Vec2], factor: double) -> Optional[Vec2]:
    if o is None:
        return None
    return Vec2(o.x * factor, o.y * factor)


def cfunc_scale(o: Optional[Vec2], factor: double) -> object:
    """
    >>> r = cfunc_scale(Vec2(2.0, 3.0), 2.0)
    >>> r is not None and abs(r.x - 4.0) < 1e-9
    True
    >>> cfunc_scale(None, 2.0) is None
    True
    """
    result: Optional[Vec2] = _cfunc_scale(o, factor)
    if result is None:
        return None
    boxed: object = result
    return boxed


# ---------------------------------------------------------------------------
# @cython.ccall with nullable param + return

@cython.ccall
def ccall_negate(o: Optional[Vec2]) -> Optional[Vec2]:
    """
    >>> r = ccall_negate(Vec2(1.0, -2.0))
    >>> r is not None and abs(r.x - (-1.0)) < 1e-9
    True
    >>> ccall_negate(None) is None
    True
    """
    if o is None:
        return None
    return Vec2(-o.x, -o.y)


# ---------------------------------------------------------------------------
# Chained calls: @cfunc result passed to @ccall

def chained(o: Optional[Vec2]) -> double:
    """
    >>> chained(Vec2(3.0, 4.0))
    100.0
    >>> chained(None)
    -1.0
    """
    scaled: Optional[Vec2] = _cfunc_scale(o, 2.0)
    negated: Optional[Vec2] = ccall_negate(scaled)
    if negated is not None:
        return negated.mag2()
    return -1.0


# ---------------------------------------------------------------------------
# Nullable param: is None / is not None dispatch

def dispatch(o: Optional[Vec2]) -> double:
    """
    >>> dispatch(Vec2(3.0, 4.0))
    3.0
    >>> dispatch(None)
    0.0
    """
    if o is None:
        return 0.0
    return o.x


# ---------------------------------------------------------------------------
# Doctest aggregator

def _doctest():
    """
    >>> def_return_some() is not None
    True
    >>> def_return_none() is None
    True
    >>> def_return_conditional(True) is not None
    True
    >>> def_return_conditional(False) is None
    True
    >>> sum_if_both(Vec2(1.0, 2.0), Vec2(3.0, 4.0)) is not None
    True
    >>> sum_if_both(None, None) is None
    True
    >>> get_x_or_default(Vec2(3.0, 4.0))
    3.0
    >>> get_x_or_default(None)
    -1.0
    >>> first_non_none() is None
    True
    >>> cfunc_scale(Vec2(2.0, 3.0), 2.0) is not None
    True
    >>> cfunc_scale(None, 2.0) is None
    True
    >>> ccall_negate(Vec2(1.0, -2.0)) is not None
    True
    >>> ccall_negate(None) is None
    True
    >>> chained(Vec2(3.0, 4.0))
    100.0
    >>> chained(None)
    -1.0
    >>> dispatch(Vec2(3.0, 4.0))
    3.0
    >>> dispatch(None)
    0.0
    """


if not cython.compiled:
    _doctest()
