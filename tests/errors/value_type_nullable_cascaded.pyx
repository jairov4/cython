# mode: error
# Cascading comparison involving a nullable value type is not supported.
# cython: language_level=3
cimport cython
from dataclasses import dataclass
from typing import Optional

@cython.value_type
@cython.final
@dataclass(frozen=True)
cdef class Vec2:
    x: cython.double
    y: cython.double

def test_cascaded_is_none(a: Optional[Vec2], b: Optional[Vec2]):
    return a is None is b

_ERRORS = """
16:13: Cascading comparison not supported for nullable value types
"""
