# mode: compile

# Python import (not cimport) of a cdef class with python_subclassing=False.
# Cython must NOT follow the Python import to load the pxd/pyx — doing so
# would be treating a runtime import as a compile-time cimport.
# The runtime __init_subclass__ on the base class covers enforcement.
# This file must compile without any errors or warnings about python_subclassing.

from python_subclassing_pyimport_base import Base

class PyChild(Base):
    pass
