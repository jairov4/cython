# mode: run
# cython: language_level=3

"""
Regression test: a @value_type classmethod referenced as a value (assigned to a
Callable / object, possibly via ``x or Cls.method``) must produce a *bound*
classmethod, not the unbound C-method pointer (whose cfunc->py wrapper would
still require the ``cls`` argument).

Reproduces a crash + runtime TypeError seen in a real project:
    TypeError: wrap() takes exactly 1 positional argument (0 given)
"""

import cython
from dataclasses import dataclass
from typing import Callable


@cython.value_type
@cython.final
@cython.cclass
@dataclass(frozen=True)
class DT:
    x: cython.long

    @classmethod
    def now(cls) -> 'DT':
        return DT(42)

    @classmethod
    def of(cls, v: cython.long) -> 'DT':
        return DT(v)


def direct() -> object:
    """Assign a value_type classmethod directly to a Python object, then call it."""
    f: object = DT.now
    return f()


def via_or(p) -> object:
    """The ``x or Cls.method`` pattern (BoolBinop coercion of the classmethod)."""
    f: object = p or DT.now
    return f()


@cython.cclass
class Host:
    _provider: object  # Callable[[], DT]

    def __init__(self, p: "Callable[[], DT] | None" = None) -> None:
        self._provider = p or DT.now

    def call_provider(self) -> object:
        return self._provider()


def with_args() -> object:
    """Classmethod taking extra args, used as a value."""
    f: object = DT.of
    return f(7)


def test_value_type_classmethod_callable():
    """
    >>> direct().x
    42
    >>> via_or(None).x
    42
    >>> via_or(lambda: DT(1)).x
    1
    >>> Host().call_provider().x
    42
    >>> Host(lambda: DT(5)).call_provider().x
    5
    >>> with_args().x
    7
    >>> isinstance(direct(), DT)
    True
    """
