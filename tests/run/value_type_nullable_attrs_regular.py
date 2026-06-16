# mode: run
# cython: language_level=3
"""
Nullable value type as attribute of a REGULAR (non-cclass) Python class and of
a @cclass.

Covers:
 - Regular Python class with Optional[Vec2] attribute: set/get/None
 - @cclass with Optional[Vec2] attribute: set/get/None and reassignment
 - Reassignment works correctly (old value released, new value acquired)

NOTE: The existing tests/run/value_type_nullable_attrs.py covers guarded/
narrowed attribute ACCESS on a nullable param. This file focuses on the attribute
STORAGE pattern (nullable as an *attribute* of another class).

NOTE: In-depth refcount assertions are not included here because they require
the compiled extension to be loaded with refnanny, which is environment-specific.
See value_type_object_fields.py for the refcount testing pattern.
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


# ---------------------------------------------------------------------------
# Regular Python class with Optional[Vec2] attribute

class RegularHolder:
    """A plain Python class; attribute typing is just a hint."""
    pos: Optional[Vec2]

    def __init__(self, pos: Optional[Vec2]) -> None:
        self.pos = pos

    def get_x(self) -> object:
        local: Optional[Vec2] = self.pos
        if local is not None:
            return local.x
        return None


def test_regular_class_none():
    """
    >>> test_regular_class_none()
    """
    h = RegularHolder(None)
    assert h.pos is None
    assert h.get_x() is None


def test_regular_class_some():
    """
    >>> test_regular_class_some()
    """
    h = RegularHolder(Vec2(3.0, 4.0))
    assert h.pos is not None
    x = h.get_x()
    assert x is not None and abs(x - 3.0) < 1e-9, x


def test_regular_class_reassign():
    """
    >>> test_regular_class_reassign()
    """
    h = RegularHolder(Vec2(1.0, 0.0))
    assert h.pos is not None

    h.pos = None
    assert h.pos is None
    assert h.get_x() is None

    h.pos = Vec2(5.0, 6.0)
    assert h.pos is not None
    x = h.get_x()
    assert x is not None and abs(x - 5.0) < 1e-9, x


# ---------------------------------------------------------------------------
# @cclass with Optional[Vec2] attribute

@cclass
class CHolder:
    pos: Optional[Vec2]

    def __init__(self, pos: Optional[Vec2]) -> None:
        self.pos = pos

    def get_x(self) -> double:
        if self.pos is not None:
            return self.pos.x
        return -1.0

    def set_pos(self, pos: Optional[Vec2]) -> None:
        self.pos = pos


def test_cclass_attr_none():
    """
    >>> test_cclass_attr_none()
    """
    c = CHolder(None)
    assert c.pos is None
    assert abs(c.get_x() - (-1.0)) < 1e-9


def test_cclass_attr_some():
    """
    >>> test_cclass_attr_some()
    """
    c = CHolder(Vec2(2.0, 3.0))
    assert c.pos is not None
    assert abs(c.get_x() - 2.0) < 1e-9, c.get_x()


def test_cclass_attr_reassign():
    """
    >>> test_cclass_attr_reassign()
    """
    c = CHolder(Vec2(1.0, 0.0))
    assert c.pos is not None

    c.set_pos(None)
    assert c.pos is None
    assert abs(c.get_x() - (-1.0)) < 1e-9

    c.set_pos(Vec2(7.0, 8.0))
    assert c.pos is not None
    assert abs(c.get_x() - 7.0) < 1e-9


def test_cclass_multiple():
    """Multiple @cclass instances with different optional states."""
    a = CHolder(Vec2(1.0, 2.0))
    b = CHolder(None)
    c = CHolder(Vec2(3.0, 4.0))

    assert a.pos is not None
    assert b.pos is None
    assert c.pos is not None

    assert abs(a.get_x() - 1.0) < 1e-9
    assert abs(b.get_x() - (-1.0)) < 1e-9
    assert abs(c.get_x() - 3.0) < 1e-9


# ---------------------------------------------------------------------------
# @cclass with Optional[Vec2] as part of a more complex pattern

@cclass
class SwappableHolder:
    """Can swap between None and non-None."""
    current: Optional[Vec2]
    previous: Optional[Vec2]

    def __init__(self) -> None:
        self.current = None
        self.previous = None

    def push(self, v: Optional[Vec2]) -> None:
        self.previous = self.current
        self.current = v

    def pop(self) -> object:
        result: Optional[Vec2] = self.current
        self.current = self.previous
        self.previous = None
        if result is None:
            return None
        boxed: object = result
        return boxed


def test_swappable_holder():
    """
    >>> test_swappable_holder()
    """
    h = SwappableHolder()
    assert h.current is None
    assert h.previous is None

    h.push(Vec2(1.0, 2.0))
    assert h.current is not None

    h.push(None)
    assert h.current is None
    assert h.previous is not None  # old current

    top = h.pop()
    assert top is None

    prev = h.pop()
    assert prev is not None and isinstance(prev, Vec2)


# ---------------------------------------------------------------------------
# Doctest aggregator

def _doctest():
    """
    >>> test_regular_class_none()
    >>> test_regular_class_some()
    >>> test_regular_class_reassign()
    >>> test_cclass_attr_none()
    >>> test_cclass_attr_some()
    >>> test_cclass_attr_reassign()
    >>> test_cclass_multiple()
    >>> test_swappable_holder()
    """


if not cython.compiled:
    _doctest()
