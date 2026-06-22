cimport cython
from cython cimport typeof

def call2():
    """
    >>> call2()
    """
    b(1,2)

def call3():
    """
    >>> call3()
    """
    b(1,2,3)

def call4():
    """
    >>> call4()
    """
    b(1,2,3,4)

# the called function:

cdef b(a, b, c=1, d=2):
    pass


cdef int foo(int a, int b=1, int c=1):
    return a+b*c

def test_foo():
    """
    >>> test_foo()
    2
    3
    7
    26
    """
    print foo(1)
    print foo(1, 2)
    print foo(1, 2, 3)
    print foo(1, foo(2, 3), foo(4))

cdef class A:
    cpdef method(self):
        """
        >>> A().method()
        'A'
        """
        return typeof(self)

cdef class B(A):
    cpdef method(self, int x = 0):
        """
        >>> B().method()
        ('B', 0)
        >>> B().method(100)
        ('B', 100)
        """
        return typeof(self), x

cdef class C(B):
    cpdef method(self, int x = 10):
        """
        >>> C().method()
        ('C', 10)
        >>> C().method(100)
        ('C', 100)
        """
        return typeof(self), x


# Sparse optional arguments: keyword calls that skip leading optionals are
# lowered to C calls that set only the matching bits of the opt-args bitmask
# (no Python dispatch).  The absence of a //GeneralCallNode proves the C path.
cdef tuple sparse_opt(int a=10, int b=20, int c=30, int d=40, int e=50):
    return (a, b, c, d, e)

@cython.test_fail_if_path_exists("//GeneralCallNode")
def test_sparse_kwargs():
    """
    >>> test_sparse_kwargs()
    [(10, 20, 30, 40, 50), (10, 20, 99, 40, 50), (10, 20, 30, 40, 7), (1, 20, 30, 4, 50), (10, 5, 30, 40, 9)]
    """
    return [
        sparse_opt(),
        sparse_opt(c=99),
        sparse_opt(e=7),
        sparse_opt(1, d=4),
        sparse_opt(b=5, e=9),
    ]
