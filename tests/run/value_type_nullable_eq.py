# mode: run
# cython: language_level=3
"""
Tests for == and != between nullable value types and bare value types,
and between two nullable value types.
"""
import cython
from cython import cclass, final, value_type, double
from dataclasses import dataclass
from typing import Optional


@value_type
@final
@cclass
@dataclass(frozen=True)
class Size:
    w: double
    h: double


# ---- nullable == bare value ----

def nullable_eq_value_true():
    """
    >>> nullable_eq_value_true()
    True
    """
    opt: Optional[Size] = Size(3.0, 4.0)
    v: Size = Size(3.0, 4.0)
    return opt == v


def nullable_eq_value_false():
    """
    >>> nullable_eq_value_false()
    False
    """
    opt: Optional[Size] = Size(1.0, 2.0)
    v: Size = Size(3.0, 4.0)
    return opt == v


def none_eq_value():
    """
    >>> none_eq_value()
    False
    """
    opt: Optional[Size] = None
    v: Size = Size(3.0, 4.0)
    return opt == v


def value_eq_nullable_true():
    """
    >>> value_eq_nullable_true()
    True
    """
    opt: Optional[Size] = Size(3.0, 4.0)
    v: Size = Size(3.0, 4.0)
    return v == opt


def value_eq_none():
    """
    >>> value_eq_none()
    False
    """
    opt: Optional[Size] = None
    v: Size = Size(3.0, 4.0)
    return v == opt


# ---- nullable != bare value ----

def nullable_ne_value_false():
    """
    >>> nullable_ne_value_false()
    False
    """
    opt: Optional[Size] = Size(3.0, 4.0)
    v: Size = Size(3.0, 4.0)
    return opt != v


def none_ne_value():
    """
    >>> none_ne_value()
    True
    """
    opt: Optional[Size] = None
    v: Size = Size(3.0, 4.0)
    return opt != v


def value_ne_none():
    """
    >>> value_ne_none()
    True
    """
    opt: Optional[Size] = None
    v: Size = Size(3.0, 4.0)
    return v != opt


# ---- nullable == nullable ----

def none_eq_none():
    """
    >>> none_eq_none()
    True
    """
    a: Optional[Size] = None
    b: Optional[Size] = None
    return a == b


def none_eq_some():
    """
    >>> none_eq_some()
    False
    """
    a: Optional[Size] = None
    b: Optional[Size] = Size(1.0, 2.0)
    return a == b


def some_eq_none():
    """
    >>> some_eq_none()
    False
    """
    a: Optional[Size] = Size(1.0, 2.0)
    b: Optional[Size] = None
    return a == b


def some_eq_same():
    """
    >>> some_eq_same()
    True
    """
    a: Optional[Size] = Size(3.0, 4.0)
    b: Optional[Size] = Size(3.0, 4.0)
    return a == b


def some_eq_diff():
    """
    >>> some_eq_diff()
    False
    """
    a: Optional[Size] = Size(3.0, 4.0)
    b: Optional[Size] = Size(5.0, 6.0)
    return a == b


def none_ne_none():
    """
    >>> none_ne_none()
    False
    """
    a: Optional[Size] = None
    b: Optional[Size] = None
    return a != b


def some_ne_same():
    """
    >>> some_ne_same()
    False
    """
    a: Optional[Size] = Size(3.0, 4.0)
    b: Optional[Size] = Size(3.0, 4.0)
    return a != b


def some_ne_diff():
    """
    >>> some_ne_diff()
    True
    """
    a: Optional[Size] = Size(3.0, 4.0)
    b: Optional[Size] = Size(5.0, 6.0)
    return a != b


# ---- via cclass attribute ----

@cython.cclass
class Container:
    cached: Optional[Size]

    def __init__(self, w: double, h: double, is_none: cython.bint):
        if is_none:
            self.cached = None
        else:
            self.cached = Size(w, h)

    def eq_value(self, v: Size) -> cython.bint:
        return self.cached == v

    def ne_value(self, v: Size) -> cython.bint:
        return self.cached != v

    def eq_nullable(self, other: Optional[Size]) -> cython.bint:
        return self.cached == other


def test_cclass_cached_none_eq_value():
    """
    >>> test_cclass_cached_none_eq_value()
    False
    """
    c = Container(0.0, 0.0, True)
    return c.eq_value(Size(3.0, 4.0))


def test_cclass_cached_value_eq_value_true():
    """
    >>> test_cclass_cached_value_eq_value_true()
    True
    """
    c = Container(3.0, 4.0, False)
    return c.eq_value(Size(3.0, 4.0))


def test_cclass_cached_value_eq_value_false():
    """
    >>> test_cclass_cached_value_eq_value_false()
    False
    """
    c = Container(3.0, 4.0, False)
    return c.eq_value(Size(1.0, 2.0))


def test_cclass_none_ne_value():
    """
    >>> test_cclass_none_ne_value()
    True
    """
    c = Container(0.0, 0.0, True)
    return c.ne_value(Size(3.0, 4.0))


def test_cclass_nullable_eq_nullable_both_none():
    """
    >>> test_cclass_nullable_eq_nullable_both_none()
    True
    """
    c = Container(0.0, 0.0, True)
    other: Optional[Size] = None
    return c.eq_nullable(other)


def test_cclass_nullable_eq_nullable_some_same():
    """
    >>> test_cclass_nullable_eq_nullable_some_same()
    True
    """
    c = Container(3.0, 4.0, False)
    other: Optional[Size] = Size(3.0, 4.0)
    return c.eq_nullable(other)


def test_cclass_nullable_eq_nullable_some_diff():
    """
    >>> test_cclass_nullable_eq_nullable_some_diff()
    False
    """
    c = Container(3.0, 4.0, False)
    other: Optional[Size] = Size(1.0, 2.0)
    return c.eq_nullable(other)
