# mode: run
# cython: language_level=3
# tag: no-cpp

"""
Tests for value_type + freelist with object fields.

Verifies that the freelist path does not corrupt or leak refcounts when a
@value_type class has PyObject fields and also uses @freelist(N).
"""

import cython
import sys
from dataclasses import dataclass


@cython.cclass
class Obj:
    v: cython.int

    def __init__(self, v: cython.int) -> None:
        self.v = v


@cython.value_type
@cython.final
@cython.freelist(64)
@cython.cclass
@dataclass(frozen=True)
class VT:
    """Value type with two object fields and a boolean, plus freelist(64)."""
    a: Obj
    b: Obj
    flag: cython.bint


@cython.value_type
@cython.final
@cython.cclass
@dataclass(frozen=True)
class Inner:
    """Inner value type with an object field (no freelist)."""
    obj: Obj
    x: cython.int


@cython.value_type
@cython.final
@cython.freelist(64)
@cython.cclass
@dataclass(frozen=True)
class Outer:
    """Outer value type with a nested value type field plus freelist(64)."""
    inner: Inner
    y: cython.int


def _rc(o: object) -> cython.int:
    """sys.getrefcount minus 1 (subtract the getrefcount argument itself)."""
    return sys.getrefcount(o) - 1


def test_vt_freelist_no_crash():
    """
    Repeated alloc/dealloc of a freelist-backed value type with object fields
    must not crash or corrupt memory.

    >>> test_vt_freelist_no_crash()
    True
    """
    o1 = Obj(1)
    o2 = Obj(2)
    for _ in range(100000):
        boxed: object = VT(o1, o2, True)
        del boxed
    return True


def test_vt_freelist_refcount():
    """
    Refcounts on the object fields must return to their baseline after churn.

    >>> test_vt_freelist_refcount()
    True
    """
    if not cython.compiled:
        return True   # refcount assertion only meaningful in compiled mode
    o1 = Obj(1)
    o2 = Obj(2)
    base1 = _rc(o1)
    base2 = _rc(o2)
    for _ in range(100000):
        boxed: object = VT(o1, o2, True)
        del boxed
    assert _rc(o1) == base1, (_rc(o1), base1)
    assert _rc(o2) == base2, (_rc(o2), base2)
    return True


def test_outer_freelist_no_crash():
    """
    Outer value type with nested value-type field (which itself has object
    fields) plus freelist must not crash.

    >>> test_outer_freelist_no_crash()
    True
    """
    o = Obj(42)
    for _ in range(100000):
        inner = Inner(o, 7)
        boxed: object = Outer(inner, 99)
        del boxed
    return True


def test_outer_freelist_refcount():
    """
    Refcount of object held inside a nested value type must return to baseline.
    We delete the 'inner' local before checking so only __pyx_v_o holds a ref.

    >>> test_outer_freelist_refcount()
    True
    """
    if not cython.compiled:
        return True
    o = Obj(42)
    base = _rc(o)
    for _ in range(100000):
        inner = Inner(o, 7)
        boxed: object = Outer(inner, 99)
        del inner
        del boxed
    assert _rc(o) == base, (_rc(o), base)
    return True


def test_nullable_vt_assignment_no_crash():
    """
    Assigning a value type to a nullable (T | None) variable in a loop must not
    crash or corrupt memory.  This exercises the NullableValueCoercionNode path
    where the inner value struct is copied and refcounts must be correctly managed.

    >>> test_nullable_vt_assignment_no_crash()
    True
    """
    o1 = Obj(1)
    o2 = Obj(2)
    opt: VT | None = None
    for _ in range(100000):
        opt = VT(o1, o2, True)   # VT(...) coerced directly to nullable
        opt = None
    return True


def test_nullable_vt_assignment_refcount():
    """
    Refcounts on object fields stay at baseline when a value type is repeatedly
    assigned to and released from a nullable (T | None) variable.

    >>> test_nullable_vt_assignment_refcount()
    True
    """
    if not cython.compiled:
        return True
    o1 = Obj(1)
    o2 = Obj(2)
    base1 = _rc(o1)
    base2 = _rc(o2)
    opt: VT | None = None
    for _ in range(100000):
        opt = VT(o1, o2, True)   # VT(...) coerced directly to nullable
        opt = None
    assert _rc(o1) == base1, (_rc(o1), base1)
    assert _rc(o2) == base2, (_rc(o2), base2)
    return True


def test_nullable_vt_loop_reassign_refcount():
    """
    A nullable variable that is re-assigned on each loop iteration must properly
    DECREF the old value before taking the new one (no leaks).

    >>> test_nullable_vt_loop_reassign_refcount()
    True
    """
    if not cython.compiled:
        return True
    o1 = Obj(1)
    o2 = Obj(2)
    base1 = _rc(o1)
    base2 = _rc(o2)
    opt: VT | None = None
    for _ in range(100000):
        opt = VT(o1, o2, True)   # reassign nullable on each iteration
    opt = None
    assert _rc(o1) == base1, (_rc(o1), base1)
    assert _rc(o2) == base2, (_rc(o2), base2)
    return True


def _doctest():
    import doctest
    import sys
    results = doctest.testmod(sys.modules[__name__])
    if results.failed:
        raise RuntimeError(f"{results.failed} doctest(s) failed")


if __name__ == '__main__':
    _doctest()
