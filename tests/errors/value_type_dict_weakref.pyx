# mode: error
cimport cython
from dataclasses import dataclass

@cython.value_type
@cython.final
@dataclass(frozen=True)
cdef class Vec:
    x: cython.double
    cdef dict __dict__


_ERRORS = """
5:0: value_type classes cannot have __dict__ or __weakref__
"""
