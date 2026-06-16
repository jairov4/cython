# mode: run
# cython: language_level=3
"""
Narrowing of nullable value types inside None-guard blocks.

Tests that:
 - un-narrowed access on None raises AttributeError
 - `if x is not None:` branch allows access without raising
 - `assert x is not None` then access allows access without raising
 - `if x is None: <early return>` (inverted guard) also permits access after
 - Nested narrowing: nullable inside another nullable branch
 - Method calls on narrowed optional work correctly
 - Narrowing does not persist outside the guarded block
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

    def scale(self, f: double) -> 'Vec2':
        return Vec2(self.x * f, self.y * f)


# ---------------------------------------------------------------------------
# Guarded access: None raises AttributeError

def access_x_unguarded(o: Optional[Vec2]) -> double:
    """Access .x without a None check — raises AttributeError if None."""
    return o.x


def test_unguarded_raises_on_none():
    """
    >>> test_unguarded_raises_on_none()
    """
    try:
        access_x_unguarded(None)
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert "NoneType" in str(e) or "None" in str(e), str(e)


def test_unguarded_works_on_value():
    """
    >>> test_unguarded_works_on_value()
    """
    assert abs(access_x_unguarded(Vec2(3.0, 4.0)) - 3.0) < 1e-9


# ---------------------------------------------------------------------------
# if-is-not-None narrowing

def get_x_if_guard(o: Optional[Vec2]) -> double:
    if o is not None:
        return o.x    # narrowed: no AttributeError guard in generated C
    return -1.0


def test_if_guard_some():
    """
    >>> test_if_guard_some()
    """
    assert abs(get_x_if_guard(Vec2(5.0, 6.0)) - 5.0) < 1e-9


def test_if_guard_none():
    """
    >>> test_if_guard_none()
    """
    assert abs(get_x_if_guard(None) - (-1.0)) < 1e-9


# ---------------------------------------------------------------------------
# assert-is-not-None narrowing

def get_x_assert_guard(o: Optional[Vec2]) -> double:
    assert o is not None, "expected non-None"
    return o.x    # narrowed after assert


def test_assert_guard():
    """
    >>> test_assert_guard()
    """
    assert abs(get_x_assert_guard(Vec2(7.0, 0.0)) - 7.0) < 1e-9


# ---------------------------------------------------------------------------
# Inverted guard: if x is None: return early

def get_x_inverted_guard(o: Optional[Vec2]) -> double:
    if o is None:
        return -1.0
    # After the early return, o is narrowed to non-None
    return o.x


def test_inverted_guard_some():
    """
    >>> test_inverted_guard_some()
    """
    assert abs(get_x_inverted_guard(Vec2(9.0, 0.0)) - 9.0) < 1e-9


def test_inverted_guard_none():
    """
    >>> test_inverted_guard_none()
    """
    assert abs(get_x_inverted_guard(None) - (-1.0)) < 1e-9


# ---------------------------------------------------------------------------
# Method call in narrowed block

def get_mag_if_guard(o: Optional[Vec2]) -> double:
    if o is not None:
        return o.mag2()   # method call on narrowed optional
    return 0.0


def test_method_in_narrowed():
    """
    >>> test_method_in_narrowed()
    """
    assert abs(get_mag_if_guard(Vec2(3.0, 4.0)) - 25.0) < 1e-9
    assert abs(get_mag_if_guard(None) - 0.0) < 1e-9


# ---------------------------------------------------------------------------
# Chained narrowed access

def chained_narrowed(o: Optional[Vec2]) -> double:
    if o is not None:
        s: Vec2 = o.scale(2.0)
        return s.mag2()
    return -1.0


def test_chained_narrowed():
    """
    >>> test_chained_narrowed()
    """
    # Vec2(3.0, 4.0).scale(2.0) = Vec2(6.0, 8.0); mag2 = 36+64 = 100
    assert abs(chained_narrowed(Vec2(3.0, 4.0)) - 100.0) < 1e-9
    assert abs(chained_narrowed(None) - (-1.0)) < 1e-9


# ---------------------------------------------------------------------------
# Narrowing doesn't extend outside the guarded block

def narrowing_scope(o: Optional[Vec2]) -> object:
    """
    Access inside guard is narrowed; access outside would require a new guard.
    """
    result = -1.0
    if o is not None:
        result = o.x    # narrowed
    # After the if-block, o is again nullable
    # (Don't access o.x here without a guard)
    return result


def test_narrowing_scope():
    """
    >>> test_narrowing_scope()
    """
    assert abs(narrowing_scope(Vec2(4.0, 5.0)) - 4.0) < 1e-9
    assert abs(narrowing_scope(None) - (-1.0)) < 1e-9


# ---------------------------------------------------------------------------
# or-None / and-None boolean patterns

def get_x_or_none_pattern(o: Optional[Vec2]) -> object:
    """Return x if non-None, else return Python None."""
    if o is None:
        return None
    return o.x  # narrowed


def test_or_none_pattern():
    """
    >>> test_or_none_pattern()
    """
    r = get_x_or_none_pattern(Vec2(2.0, 3.0))
    assert r is not None and abs(r - 2.0) < 1e-9
    assert get_x_or_none_pattern(None) is None


# ---------------------------------------------------------------------------
# Doctest aggregator

def _doctest():
    """
    >>> test_unguarded_raises_on_none()
    >>> test_unguarded_works_on_value()
    >>> test_if_guard_some()
    >>> test_if_guard_none()
    >>> test_assert_guard()
    >>> test_inverted_guard_some()
    >>> test_inverted_guard_none()
    >>> test_method_in_narrowed()
    >>> test_chained_narrowed()
    >>> test_narrowing_scope()
    >>> test_or_none_pattern()
    """


if not cython.compiled:
    _doctest()
