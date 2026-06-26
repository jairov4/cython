# mode: run
# cython: language_level=3

"""
Regression test: a value_type with PyObject fields captured into a closure
(or generator) scope must keep the scope GC-aware, so tp_traverse/tp_clear
visit the embedded PyObject fields.  Otherwise the GC cannot see those
references and may collect them prematurely (use-after-free) or fail to
collect reference cycles routed through the closure (leak).

This reproduces a crash seen in a real project where a value_type with a
cclass field was captured by a lambda held alive by a generator.
"""

import cython
import gc
import weakref
from dataclasses import dataclass


@cython.cclass
class Obj:
    tag: object

    def __init__(self, tag: object) -> None:
        self.tag = tag

    def get_tag(self):
        return self.tag


@cython.value_type
@cython.final
@cython.cclass
@dataclass(frozen=True)
class VT:
    a: Obj
    flag: cython.bint


def _make_capturing_closure(o: Obj):
    # The lambda captures the value_type local `v` (a struct embedding the
    # PyObject field `a`).  The closure scope object thus holds a reference to
    # `o` only through the embedded value struct.
    v: VT = VT(o, True)
    return lambda: v.a


def test_closure_keeps_field_alive():
    """The captured value_type's object field stays alive while the closure does."""
    o = Obj("keep")
    fn = _make_capturing_closure(o)
    del o
    gc.collect()
    # If the closure scope weren't GC-aware / didn't own the field, `o` could
    # be freed here -> accessing it would crash.
    got = fn()
    assert got.get_tag() == "keep", got
    return True


class Marker:
    """Pure-Python (weakref-able) cycle member."""
    pass


def test_closure_cycle_is_collectable():
    """
    A reference cycle routed through the closure scope's embedded value_type
    must be collectable.  If the closure scope lacks tp_traverse/tp_clear for
    the embedded PyObject field, the cycle leaks (uncollectable).
    """
    if not cython.compiled:
        return True  # GC-tracking specifics only meaningful when compiled

    m = Marker()
    o = Obj(m)                       # Obj.tag -> Marker
    fn = _make_capturing_closure(o)  # closure scope embeds VT(o); scope -> o
    m.fn = fn                        # Marker -> fn -> scope -> v.a == o -> o.tag == m  (cycle)
    w = weakref.ref(m)
    del m, o, fn
    gc.collect()
    # The Marker must be collected; a missing tp_traverse/tp_clear on the
    # closure scope would leave this cycle uncollectable.
    assert w() is None, "cycle leaked: closure scope not GC-traversing its value_type field"
    return True


def _doctest():
    """
    >>> test_closure_keeps_field_alive()
    True
    >>> test_closure_cycle_is_collectable()
    True
    """
