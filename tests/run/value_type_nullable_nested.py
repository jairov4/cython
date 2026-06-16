# mode: run
# cython: language_level=3
"""
Nullable value types stored in locals, tested for nesting patterns.

Tests:
 - Optional[Inner] used as a local variable
 - A non-value-type class (cclass) with an Optional[Vec2] attribute
 - Passing Optional[Inner] as a param and extracting .a inside a guard
 - Constructing an inner-optional value and extracting its field via narrowing
 - GC / refcount for a value class that itself has an Optional[Sentinel] field
   (optional payload carrying a Python object, in a @cclass container)
 - Embedding Optional[Inner] as a *field* of another value_type class
   (Bug 2 regression: struct forward-declaration ordering + C++ operator==)
"""

import cython
import sys
from cython import cclass, final, value_type, double
from dataclasses import dataclass
from typing import Optional


@value_type
@final
@cclass
@dataclass(frozen=True)
class Inner:
    a: double


# A value_type whose field is Optional[Inner] — Bug 2 regression test.
# The compiler previously emitted __pyx_val_Outer before __pyx_optval_Inner
# (incomplete type) and lacked operator== on the optional struct.
@value_type
@final
@cclass
@dataclass(frozen=True)
class Outer:
    maybe: Optional[Inner]
    flag: double


# A cclass (NOT a value_type) that holds Optional[Inner] attributes.
@cclass
class Box:
    inner: Optional[Inner]

    def __init__(self, inner: Optional[Inner]) -> None:
        self.inner = inner

    def get_a(self) -> double:
        if self.inner is not None:
            return self.inner.a
        return -1.0


# A value_type whose object-field is Optional[object] (a regular Python object,
# not another value type), to test GC through a nullable Python-object field.
@cython.cclass
class Sentinel:
    value: cython.int

    def __init__(self, value: cython.int) -> None:
        self.value = value


@value_type
@final
@cclass
@dataclass(frozen=True)
class WithOptObj:
    """Value type with an Optional[Sentinel] (cclass) field — tests GC path."""
    n: double
    obj: Optional[Sentinel]


def _rc(o: object) -> cython.int:
    return sys.getrefcount(o) - 1


# ---------------------------------------------------------------------------
# Optional[Inner] as a local variable

def test_local_none():
    """Local Optional[Inner] = None."""
    x: Optional[Inner] = None
    assert x is None


def test_local_some():
    """Local Optional[Inner] = Inner(...)."""
    x: Optional[Inner] = Inner(3.0)
    assert x is not None
    if x is not None:
        assert abs(x.a - 3.0) < 1e-9


def test_local_reassign():
    """Reassign Optional[Inner] local from None to Some and back."""
    x: Optional[Inner] = None
    assert x is None
    x = Inner(5.0)
    assert x is not None
    if x is not None:
        assert abs(x.a - 5.0) < 1e-9
    x = None
    assert x is None


# ---------------------------------------------------------------------------
# @cfunc using Optional[Inner]

@cython.cfunc
def _scale_inner(o: Optional[Inner], factor: double) -> Optional[Inner]:
    if o is None:
        return None
    return Inner(o.a * factor)


def test_cfunc_optional_inner():
    """@cfunc returning Optional[Inner] and chaining."""
    r1: Optional[Inner] = _scale_inner(None, 2.0)
    assert r1 is None

    r2: Optional[Inner] = _scale_inner(Inner(3.0), 2.0)
    assert r2 is not None
    if r2 is not None:
        assert abs(r2.a - 6.0) < 1e-9

    # Chain: scale twice
    r3: Optional[Inner] = _scale_inner(r2, 0.5)
    assert r3 is not None
    if r3 is not None:
        assert abs(r3.a - 3.0) < 1e-9


# ---------------------------------------------------------------------------
# cclass attribute holding Optional[Inner]

def test_cclass_attr_none():
    """Box.inner = None."""
    b = Box(None)
    assert b.inner is None
    assert abs(b.get_a() - (-1.0)) < 1e-9


def test_cclass_attr_some():
    """Box.inner = Inner(7.0)."""
    b = Box(Inner(7.0))
    assert b.inner is not None
    assert abs(b.get_a() - 7.0) < 1e-9


def test_cclass_attr_reassign():
    """Reassign Box.inner from Some to None."""
    b = Box(Inner(1.0))
    assert b.inner is not None
    b.inner = None
    assert b.inner is None
    b.inner = Inner(2.0)
    assert b.inner is not None


# ---------------------------------------------------------------------------
# Optional[Sentinel] inside a value_type (Python object payload)

def test_with_opt_obj_none():
    """WithOptObj with optional obj=None."""
    v: WithOptObj = WithOptObj(1.0, None)
    assert v.n == 1.0
    assert v.obj is None


def test_with_opt_obj_some():
    """WithOptObj with a Sentinel payload."""
    s = Sentinel(42)
    v: WithOptObj = WithOptObj(2.0, s)
    assert v.n == 2.0
    assert v.obj is s


def test_with_opt_obj_refcount():
    """Refcount of Sentinel payload tracked through WithOptObj."""
    if not cython.compiled:
        return

    s = Sentinel(7)
    base = _rc(s)

    v: WithOptObj = WithOptObj(1.0, s)
    assert _rc(s) == base + 1, "value should hold +1 ref"

    del v
    assert _rc(s) == base, "ref released after del"


def test_with_opt_obj_gc():
    """GC cycle through boxed WithOptObj with object payload."""
    import gc
    lst = []
    v: WithOptObj = WithOptObj(0.0, None)
    boxed: object = v
    lst.append(boxed)
    del v
    del boxed
    del lst
    gc.collect()  # Should not crash or leak


# ---------------------------------------------------------------------------
# Outer: value_type embedding Optional[Inner] field (Bug 2 regression)

def test_outer_none_field():
    """Outer with maybe=None."""
    o: Outer = Outer(None, 1.0)
    assert o.maybe is None
    assert abs(o.flag - 1.0) < 1e-9


def test_outer_some_field():
    """Outer with maybe=Inner(3.0)."""
    o: Outer = Outer(Inner(3.0), 2.0)
    assert o.maybe is not None
    if o.maybe is not None:
        assert abs(o.maybe.a - 3.0) < 1e-9
    assert abs(o.flag - 2.0) < 1e-9


def test_outer_equality():
    """Outer equality via Python __eq__ through the nullable field."""
    # Use object-typed variables so the comparison goes through Python's __eq__,
    # matching the convention in value_type_nested.py (C structs have no == in C mode).
    o1: object = Outer(None, 1.0)
    o2: object = Outer(None, 1.0)
    o3: object = Outer(Inner(3.0), 1.0)
    o4: object = Outer(Inner(3.0), 1.0)
    o5: object = Outer(Inner(4.0), 1.0)
    assert o1 == o2
    assert o3 == o4
    assert o1 != o3
    assert o3 != o5


def test_outer_box_unbox():
    """Box Outer to Python and read it back."""
    o: Outer = Outer(Inner(7.0), 5.0)
    boxed: object = o
    assert isinstance(boxed, Outer)
    o2: object = boxed
    assert o2 == o  # Python-level eq (o is auto-boxed by the comparison)


# ---------------------------------------------------------------------------
# Doctest aggregator

def _doctest():
    """
    >>> test_local_none()
    >>> test_local_some()
    >>> test_local_reassign()
    >>> test_cfunc_optional_inner()
    >>> test_cclass_attr_none()
    >>> test_cclass_attr_some()
    >>> test_cclass_attr_reassign()
    >>> test_with_opt_obj_none()
    >>> test_with_opt_obj_some()
    >>> test_with_opt_obj_refcount()
    >>> test_with_opt_obj_gc()
    >>> test_outer_none_field()
    >>> test_outer_some_field()
    >>> test_outer_equality()
    >>> test_outer_box_unbox()
    """


if not cython.compiled:
    _doctest()
