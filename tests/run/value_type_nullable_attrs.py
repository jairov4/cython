# mode: run
# cython: language_level=3
"""
Phase 6: attribute/method access on nullable value types.

Tests guarded access (AttributeError on None) and narrowing
(guard eliminated after is-not-None check).
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
# Guarded access: None -> AttributeError


def get_x_guarded(o: Optional[Vec2]) -> double:
    """
    >>> get_x_guarded(Vec2(3.0, 4.0))
    3.0
    >>> try:
    ...     get_x_guarded(None)
    ...     raise AssertionError("expected AttributeError")
    ... except AttributeError as e:
    ...     "'NoneType' object has no attribute" in str(e)
    True
    """
    return o.x


def get_mag_guarded(o: Optional[Vec2]) -> double:
    """
    >>> get_mag_guarded(Vec2(3.0, 4.0))
    25.0
    >>> try:
    ...     get_mag_guarded(None)
    ...     raise AssertionError("expected AttributeError")
    ... except AttributeError as e:
    ...     True
    True
    """
    return o.mag2()


# ---------------------------------------------------------------------------
# Narrowed access: no guard inside `if o is not None:`


def get_x_safe(o: Optional[Vec2]) -> double:
    """
    >>> get_x_safe(Vec2(3.0, 4.0))
    3.0
    >>> get_x_safe(None)
    -1.0
    """
    if o is not None:
        return o.x      # UNGUARDED: no AttributeError check in C
    return -1.0


def get_mag_safe(o: Optional[Vec2]) -> double:
    """
    >>> get_mag_safe(Vec2(3.0, 4.0))
    25.0
    >>> get_mag_safe(None)
    -1.0
    """
    if o is not None:
        return o.mag2()  # UNGUARDED
    return -1.0


# ---------------------------------------------------------------------------
# Narrowed via assert


def get_x_assert(o: Optional[Vec2]) -> double:
    """
    >>> get_x_assert(Vec2(5.0, 0.0))
    5.0
    """
    assert o is not None
    return o.x   # UNGUARDED after assert


# ---------------------------------------------------------------------------
# Python fallback (interpreted mode)


def _doctest():
    """Run doctests when not compiled (pure-Python fallback)."""
    import doctest
    import sys
    results = doctest.testmod(sys.modules[__name__], verbose=False)
    assert results.failed == 0, results


if not cython.compiled:
    # In pure Python, the type annotation is just a hint; access always works.
    _doctest()
