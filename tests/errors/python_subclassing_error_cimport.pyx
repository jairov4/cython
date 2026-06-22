# mode: error

# Cross-module cimport: when the base class is cimported (not Python-imported)
# Cython does know its python_subclassing=False at compile time and must error.

from python_subclassing_pyimport_base cimport Base

class PyChild(Base):  # should error
    pass

_ERRORS = """
8:0: Python class 'PyChild' inherits from extension type 'Base' which has python_subclassing=False; declare it as a cdef class or add @cython.python_subclassing(True) to the base type
"""
