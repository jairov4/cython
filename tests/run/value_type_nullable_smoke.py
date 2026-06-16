# mode: run
# cython: language_level=3
"""
Smoke test: box/unbox round-trip for a nullable value type.
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


def box_none():
    """
    >>> box_none() is None
    True
    """
    a: Optional[Vec2] = None
    o: object = a
    return o


def box_val():
    """
    >>> box_val()
    True
    """
    a: Optional[Vec2] = Vec2(3.0, 4.0)
    o: object = a
    return o is not None and isinstance(o, Vec2)


def round_trip(o):
    """
    >>> round_trip(None) is None
    True
    >>> r = round_trip(Vec2(5.0, 6.0))
    >>> isinstance(r, Vec2)
    True
    """
    a: Optional[Vec2] = o
    o2: object = a
    return o2
