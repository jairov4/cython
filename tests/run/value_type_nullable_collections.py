# mode: run
# cython: language_level=3
"""
Nullable value type as element of Python collection types.

Covers:
 - list[Optional[Vec2]]: mixing None and values; box on insert, check is None on read
 - dict value: mapping string keys to Optional[Vec2]
 - tuple: boxing Optional[Vec2] into a Python tuple element
 - round-trip from collection back to Optional[Vec2] local
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
# list

def test_list_mixed():
    """A list of Optional[Vec2] mixing None and real values."""
    items = []
    v1: Optional[Vec2] = Vec2(1.0, 2.0)
    v2: Optional[Vec2] = None
    v3: Optional[Vec2] = Vec2(3.0, 4.0)
    items.append(v1)   # boxes to Vec2 instance
    items.append(v2)   # boxes to None
    items.append(v3)

    assert len(items) == 3
    assert items[0] is not None and isinstance(items[0], Vec2)
    assert items[1] is None
    assert items[2] is not None and isinstance(items[2], Vec2)


def test_list_read_back():
    """Read list elements back into Optional[Vec2] locals."""
    items = [Vec2(5.0, 6.0), None, Vec2(7.0, 8.0)]

    a: Optional[Vec2] = items[0]
    b: Optional[Vec2] = items[1]
    c: Optional[Vec2] = items[2]

    assert a is not None
    assert b is None
    assert c is not None
    if a is not None:
        assert abs(a.x - 5.0) < 1e-9
    if c is not None:
        assert abs(c.y - 8.0) < 1e-9


def test_list_comprehension():
    """Build list of Optional[Vec2] via comprehension."""
    vecs = [Vec2(float(i), 0.0) for i in range(3)]
    # Box each to optional and back
    opts: list = [v for v in vecs]
    for i, item in enumerate(opts):
        assert item is not None
        assert isinstance(item, Vec2)
        local: Optional[Vec2] = item
        if local is not None:
            assert abs(local.x - float(i)) < 1e-9


def test_list_count_none():
    """Count None entries in a list of Optional[Vec2]."""
    vals: list = [Vec2(0.0, 0.0), None, Vec2(1.0, 0.0), None]
    none_count = sum(1 for v in vals if v is None)
    assert none_count == 2, none_count


# ---------------------------------------------------------------------------
# dict

def test_dict_value():
    """dict mapping str -> Optional[Vec2]."""
    d: dict = {}
    a: Optional[Vec2] = Vec2(1.0, 2.0)
    d['a'] = a
    d['missing'] = None

    assert d['missing'] is None
    read_a: Optional[Vec2] = d['a']
    assert read_a is not None
    if read_a is not None:
        assert abs(read_a.x - 1.0) < 1e-9 and abs(read_a.y - 2.0) < 1e-9


def test_dict_round_trip():
    """Round-trip: set dict value, read back, check fields via narrowing."""
    d = {'p': Vec2(9.0, 3.0), 'q': None}

    p: Optional[Vec2] = d['p']
    q: Optional[Vec2] = d['q']

    assert p is not None
    assert q is None

    if p is not None:
        assert abs(p.x - 9.0) < 1e-9


# ---------------------------------------------------------------------------
# tuple

def test_tuple_element():
    """Optional[Vec2] as element of a Python tuple."""
    v: Optional[Vec2] = Vec2(2.0, 3.0)
    t = (v, None, Vec2(4.0, 5.0))

    assert t[0] is not None and isinstance(t[0], Vec2)
    assert t[1] is None
    assert t[2] is not None and isinstance(t[2], Vec2)

    read_v: Optional[Vec2] = t[0]
    if read_v is not None:
        assert abs(read_v.x - 2.0) < 1e-9


def test_tuple_unpack():
    """Unpack Optional[Vec2] tuple elements into locals."""
    a: Optional[Vec2] = Vec2(1.0, 0.0)
    b: Optional[Vec2] = None
    t = (a, b)

    x, y = t
    local_x: Optional[Vec2] = x
    local_y: Optional[Vec2] = y

    assert local_x is not None
    assert local_y is None
    if local_x is not None:
        assert abs(local_x.x - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Doctest aggregator

def _doctest():
    """
    >>> test_list_mixed()
    >>> test_list_read_back()
    >>> test_list_comprehension()
    >>> test_list_count_none()
    >>> test_dict_value()
    >>> test_dict_round_trip()
    >>> test_tuple_element()
    >>> test_tuple_unpack()
    """


if not cython.compiled:
    _doctest()
