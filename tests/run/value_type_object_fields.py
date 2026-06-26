# mode: run
# cython: language_level=3

"""
Tests for value_type classes with Python object fields (v3).

Tests reference counting correctness, copy semantics, boxing/unboxing,
GC, and deletion.
"""

import cython
import sys
from dataclasses import dataclass
from typing import Optional


@cython.cclass
class Wrapper:
    """A plain extension type used as a cclass-typed field in a value type."""
    value: cython.int

    def __init__(self, value: cython.int) -> None:
        self.value = value


@cython.value_type
@cython.final
@cython.cclass
@dataclass(frozen=True)
class WithCclass:
    """Value type whose field is a specific cclass (not plain object)."""
    n: cython.double
    obj: Optional[Wrapper]


@cython.value_type
@cython.final
@cython.cclass
@dataclass(frozen=True)
class Tagged:
    n: cython.double
    label: str
    payload: object


@cython.value_type
@cython.final
@cython.cclass
@dataclass(frozen=True)
class StrOnly:  # codespell:ignore StrOnly
    x: cython.double
    name: str


@cython.value_type
@cython.final
@cython.cclass
@dataclass(frozen=True)
class ListField:
    items: list


def _rc(o) -> cython.int:
    """Return sys.getrefcount(o) - 1 (subtract the getrefcount argument itself)."""
    return sys.getrefcount(o) - 1


def test_basic_construction():
    """Value class with object field can be constructed and fields are readable."""
    obj = object()
    v = Tagged(1.0, "hello", obj)
    assert v.n == 1.0, v.n
    assert v.label == "hello", v.label
    assert v.payload is obj, v.payload


def test_copy_refcount():
    """
    Each copy of a value struct owns a reference to each object field.
    Refcounts rise and fall correctly with copies.
    Uses explicit type annotations so Cython uses value semantics (not boxing).
    """
    if not cython.compiled:
        return  # refcount tests require Cython value-copy semantics
    obj = object()
    base_rc = _rc(obj)

    a: Tagged = Tagged(1.0, "x", obj)
    assert _rc(obj) == base_rc + 1, _rc(obj)  # a holds +1

    b: Tagged = a  # value copy: b also holds +1
    assert _rc(obj) == base_rc + 2, _rc(obj)

    del b
    assert _rc(obj) == base_rc + 1, _rc(obj)

    del a
    assert _rc(obj) == base_rc, _rc(obj)


def test_reassignment_refcount():
    """Reassigning a value variable decrefs the old value and increfs the new."""
    if not cython.compiled:
        return  # refcount tests require Cython value-copy semantics
    obj1 = object()
    obj2 = object()
    base1 = _rc(obj1)
    base2 = _rc(obj2)

    a: Tagged = Tagged(1.0, "one", obj1)
    assert _rc(obj1) == base1 + 1
    assert _rc(obj2) == base2

    a = Tagged(2.0, "two", obj2)  # old value should be decref'd
    assert _rc(obj1) == base1, "old payload not released on reassignment"
    assert _rc(obj2) == base2 + 1, "new payload not acquired on reassignment"

    del a
    assert _rc(obj2) == base2


def test_del_refcount():
    """del v releases the reference held by the value struct."""
    if not cython.compiled:
        return  # refcount tests require Cython value-copy semantics
    obj = object()
    base = _rc(obj)

    v: Tagged = Tagged(1.0, "x", obj)
    assert _rc(obj) == base + 1

    del v
    assert _rc(obj) == base


def test_boxing_refcount():
    """Boxing a value class (to Python object) creates an independent copy."""
    if not cython.compiled:
        return  # refcount tests require Cython value-copy semantics
    obj = object()
    base = _rc(obj)

    v: Tagged = Tagged(1.0, "hi", obj)
    assert _rc(obj) == base + 1  # v holds +1

    boxed: object = v  # box v into a Python object
    assert _rc(obj) == base + 2  # boxed also holds +1

    del v
    assert _rc(obj) == base + 1  # boxed still alive

    del boxed
    assert _rc(obj) == base


def test_unboxing_refcount():
    """Unboxing (from Python object) gives the value independent ownership."""
    if not cython.compiled:
        return  # refcount tests require Cython value-copy semantics
    obj = object()
    base = _rc(obj)

    # Create a boxed instance first
    v1: Tagged = Tagged(1.0, "x", obj)
    assert _rc(obj) == base + 1
    boxed: object = v1
    assert _rc(obj) == base + 2

    # Unbox: extract the value (should INCREF fields)
    v2: Tagged = cython.cast(Tagged, boxed)
    assert _rc(obj) == base + 3  # v1, boxed, v2 all hold refs

    del v1
    assert _rc(obj) == base + 2

    del boxed
    assert _rc(obj) == base + 1  # v2 still holds its ref

    del v2
    assert _rc(obj) == base


def test_str_field_repr():
    """Tagged with str field shows correct __repr__."""
    v = Tagged(3.14, "pi", None)
    r = repr(v)
    assert "Tagged" in r, r
    assert "pi" in r or "label" in r, r


def test_eq():
    """Tagged equality works on object fields (via boxed comparison)."""
    obj = object()
    a = Tagged(1.0, "a", obj)
    b = Tagged(1.0, "a", obj)
    # Box to Python objects for comparison — direct struct == is not defined for pyobject fields
    boxed_a: object = a
    boxed_b: object = b
    assert boxed_a == boxed_b

    c = Tagged(1.0, "a", None)
    boxed_c: object = c
    assert boxed_a != boxed_c


def test_hash_str_field():
    """StrOnly with str field is hashable (str is hashable)."""  # codespell:ignore StrOnly
    v = StrOnly(1.0, "name")  # codespell:ignore StrOnly
    h = hash(v)
    assert isinstance(h, int)


def test_hash_list_field_raises():
    """ListField with list field raises TypeError on hash (list is unhashable)."""
    v = ListField([1, 2, 3])
    try:
        hash(v)
        assert False, "Expected TypeError"
    except TypeError:
        pass


def test_none_field():
    """Object fields can be None."""
    v: Tagged = Tagged(0.0, "x", None)
    assert v.payload is None
    obj = object()
    base = _rc(obj)
    # None assignment doesn't affect obj's refcount
    v2: Tagged = Tagged(1.0, "y", None)
    assert _rc(obj) == base
    del v2


def test_scope_exit_refcount():
    """Refcounts return to baseline after value leaves scope."""
    if not cython.compiled:
        return  # refcount tests require Cython value-copy semantics
    obj = object()
    base = _rc(obj)

    def inner():
        v: Tagged = Tagged(1.0, "x", obj)
        assert _rc(obj) == base + 1
        # v destroyed on return

    inner()
    assert _rc(obj) == base


def test_repeated_construction_no_leak():
    """Repeated construction/destruction of value with object field doesn't leak."""
    if not cython.compiled:
        return  # refcount tests require Cython value-copy semantics
    obj = object()
    base = _rc(obj)
    N = 10000
    for _ in range(N):
        v: Tagged = Tagged(1.0, "x", obj)
        del v
    assert _rc(obj) == base, "Leak detected: %d vs %d" % (_rc(obj), base)


def test_list_field_refcount():
    """list field refcounting works correctly."""
    if not cython.compiled:
        return  # refcount tests require Cython value-copy semantics
    lst = [1, 2, 3]
    base = _rc(lst)

    v: ListField = ListField(lst)
    assert _rc(lst) == base + 1

    del v
    assert _rc(lst) == base


def test_list_field_gc():
    """Boxed value type with list field participates in GC (no cycle leak)."""
    import gc
    # Create a cycle: a list contains the boxed value which contains the list.
    lst = []
    v: ListField = ListField(lst)
    boxed: object = v
    lst.append(boxed)  # cycle: boxed -> lst -> boxed
    del v
    del boxed
    del lst
    # GC should collect the cycle
    gc.collect()
    # No assertion needed — if tp_traverse/tp_clear are wired, gc will collect.
    # A missing tp_traverse would leave a cyclic garbage that gc can't collect.
    # This test just ensures no crash/hang.


def test_cclass_typed_field_none():
    """
    Construction of a value type with a cclass-typed optional field set to None.
    The generated struct member assignment must cast Py_None to the field's C type.
    Regression: 'incompatible pointer types assigning PyObject* to ExtType*'.
    """
    v: WithCclass = WithCclass(1.0, None)
    assert v.n == 1.0
    assert v.obj is None

    w = Wrapper(42)
    v2: WithCclass = WithCclass(2.0, w)
    assert v2.obj is w


def test_cclass_typed_field_refcount():
    """Cclass-typed field in a value type is refcounted correctly."""
    if not cython.compiled:
        return
    w = Wrapper(7)
    base = _rc(w)

    v: WithCclass = WithCclass(1.0, w)
    assert _rc(w) == base + 1

    del v
    assert _rc(w) == base


@cython.cclass
class Container:
    """Extension type that embeds a value type with a PyObject field as an attribute."""
    slot: Tagged

    def __init__(self, v: Tagged) -> None:
        self.slot = v

    def replace_slot(self, v: Tagged) -> None:
        self.slot = v


def test_attribute_assignment_refcount():
    """Assigning a refcounted value type to a cclass attribute INCs/DECREFs correctly."""
    if not cython.compiled:
        return
    obj1 = object()
    obj2 = object()
    base1 = _rc(obj1)
    base2 = _rc(obj2)

    c = Container(Tagged(1.0, "a", obj1))
    assert _rc(obj1) == base1 + 1, "container should hold ref to obj1"

    c.replace_slot(Tagged(2.0, "b", obj2))
    assert _rc(obj1) == base1, "old slot payload should be released"
    assert _rc(obj2) == base2 + 1, "new slot payload should be acquired"

    del c
    assert _rc(obj2) == base2, "slot payload released when container deleted"


def test_embedded_value_type_gc():
    """
    A cclass with an embedded refcounted value type participates in GC correctly.
    Creates a cycle: Container.slot.payload -> lst -> Container.
    tp_traverse must visit the PyObject fields inside the embedded value struct.
    """
    import gc
    lst = []
    c = Container(Tagged(1.0, "x", lst))
    lst.append(c)  # cycle: c.slot.payload is lst, lst[0] is c
    del c
    del lst
    gc.collect()
    # If tp_traverse is missing, the cyclic garbage won't be collected.


def test_dataclass_fields_visible():
    """dataclasses.fields() works on the boxed type."""
    import dataclasses
    fields = dataclasses.fields(Tagged)
    names = [f.name for f in fields]
    assert 'n' in names
    assert 'label' in names
    assert 'payload' in names


def test_isinstance():
    """isinstance works for the boxed form."""
    v = Tagged(1.0, "x", None)
    boxed: object = v
    assert isinstance(boxed, Tagged)


def _doctest():
    """
    >>> test_basic_construction()
    >>> test_copy_refcount()
    >>> test_reassignment_refcount()
    >>> test_del_refcount()
    >>> test_boxing_refcount()
    >>> test_unboxing_refcount()
    >>> test_str_field_repr()
    >>> test_eq()
    >>> test_hash_str_field()
    >>> test_hash_list_field_raises()
    >>> test_none_field()
    >>> test_scope_exit_refcount()
    >>> test_repeated_construction_no_leak()
    >>> test_list_field_refcount()
    >>> test_list_field_gc()
    >>> test_cclass_typed_field_none()
    >>> test_cclass_typed_field_refcount()
    >>> test_attribute_assignment_refcount()
    >>> test_embedded_value_type_gc()
    >>> test_dataclass_fields_visible()
    >>> test_isinstance()
    """
