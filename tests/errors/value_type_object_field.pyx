# mode: compile
# Object/str/list fields in value_type classes are now allowed (v3).
# This file verifies that a value_type class with an object field compiles successfully.
cimport cython
from dataclasses import dataclass

@cython.value_type
@cython.final
@dataclass(frozen=True)
cdef class Tagged:
    n: cython.double
    label: object
