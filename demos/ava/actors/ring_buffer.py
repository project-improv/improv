#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""*Provides a numpy ring buffer at a fixed memory address to allow for
significantly sped up* ``numpy``, ``sigpy``, ``numba`` *&* ``pyFFTW``
*calculations.*

- Github: https://github.com/Dennis-van-Gils/python-dvg-ringbuffer
- PyPI: https://pypi.org/project/dvg-ringbuffer

Installation::

    pip install dvg-ringbuffer

Based on:

    https://pypi.org/project/numpy_ringbuffer/ by Eric Wieser.

    ``DvG_RingBuffer`` can be used as a drop-in replacement for
    ``numpy_ringbuffer`` and provides several optimizations and extra features,
    but requires Python 3.

If and only if the ring buffer is completely full, will it return its array
data as a contiguous C-style numpy array at a single fixed memory address per
ring buffer instance. It does so by unwrapping the discontiguous ring buffer
array into a second extra *unwrap* buffer that is a private member of the ring
buffer class. This is advantegeous for other accelerated computations by, e.g.,
``numpy``, ``sigpy``, ``numba`` & ``pyFFTW``, that benefit from being fed with
contiguous arrays at the same memory address each time again, such that compiler
optimizations and data planning are made possible.

When the ring buffer is not completely full, it will return its data as a
contiguous C-style numpy array, but at different memory addresses. This is how
the original ``numpy-buffer`` always operates.

Commonly, ``collections.deque()`` is used to act as a ring buffer. The
benefits of a deque is that it is thread safe and fast (enough) for most
situations. However, there is an overhead whenever the deque -- a list-like
container -- needs to be transformed into a numpy array. Because
``DvG_RingBuffer`` already returns numpy arrays it will outperform a
``collections.deque()`` easily, tested to be a factor of ~60.

.. warning::

    * This ring buffer is not thread safe. You'll have to implement your own
      mutex locks when using this ring buffer in multithreaded operations.

    * The data array that is returned by a full ring buffer is a pass by
      reference of the *unwrap* buffer. It is not a copy! Hence, changing
      values in the returned data array is identical to changing values in the
      *unwrap* buffer.

API
===

``class RingBuffer(capacity, dtype=float, allow_overwrite=True)``
----------------------------------------------------------------------
    Create a new ring buffer with the given capacity and element type.

        Args:
            capacity (``int``):
                The maximum capacity of the ring buffer

            dtype (``data-type``, optional):
                Desired type of buffer elements. Use a type like ``(float, 2)``
                to produce a buffer with shape ``(capacity, 2)``.

                Default: ``float``

            allow_overwrite (``bool``, optional):
                If ``False``, throw an IndexError when trying to append to an
                already full buffer.

                Default: ``True``

Methods
-------
* ``clear()``
* ``append(value)``
    Append a single value to the ring buffer.

    .. code-block:: python

        rb = RingBuffer(3, dtype=int)  #  []
        rb.append(1)                   #  [1]
        rb.append(2)                   #  [1, 2]
        rb.append(3)                   #  [1, 2, 3]
        rb.append(4)                   #  [2, 3, 4]

* ``appendleft(value)``
    Append a single value to the ring buffer from the left side.

    .. code-block:: python

        rb = RingBuffer(3, dtype=int)  #  []
        rb.appendleft(1)               #  [1]
        rb.appendleft(2)               #  [2, 1]
        rb.appendleft(3)               #  [3, 2, 1]
        rb.appendleft(4)               #  [4, 3, 2]

* ``extend(values)``
    Extend the ring buffer with a list of values.

    .. code-block:: python

        rb = RingBuffer(3, dtype=int)  #  []
        rb.extend([1])                 #  [1]
        rb.extend([2, 3])              #  [1, 2, 3]
        rb.extend([4, 5, 6, 7])        #  [5, 6, 7]

* ``extendleft(values)``
    Extend the ring buffer with a list of values from the left side.

    .. code-block:: python

        rb = RingBuffer(3, dtype=int)  #  []
        rb.extendleft([1])             #  [1]
        rb.extendleft([3, 2])          #  [3, 2, 1]
        rb.extendleft([7, 6, 5, 4])    #  [7, 6, 5]

* ``pop(n=1, n_overlap=0)``
    Remove and return n elements(s) from the right side of the ring buffer. Include n_overlap in remaining buffer.

    .. code-block:: python

        rb = RingBuffer(3, dtype=int)  # --> rb = []
        rb.extend([1, 2, 3])           # --> rb = [1, 2, 3]
        rb.pop()                       # --> rb = [1, 2], res = [3]
        rb.pop(n=2)                      # --> rb = [], res = [1, 2]

        # EXPECTED BEHAVIOR
        rb = RingBuffer(3, dtype=int)  # --> rb = []
        rb.extend([1, 2, 3])           # --> rb = [1, 2, 3]
        rb.pop(n=2, n_overlap=1)       # --> rb = [1, 2], res = [2, 3]
        rb.pop(n=2, n_overlap=1)       # --> rb = [1], res = [1, 2]
        
        TODO: use case: n=2, n_overlap=2

* ``popleft(n=1, n_overlap=0)``
    Remove and return n element(s) from the left side of the ring buffer. Include n_overlap in remaining buffer.

    .. code-block:: python

        rb = RingBuffer(3, dtype=int)  # --> rb = []
        rb.extend([1, 2, 3])           # --> rb = [1, 2, 3]
        res = rb.popleft()             # --> rb = [2, 3], res = [1]
        res = rb.popleft(n=2)            # --> rb = [1], res = [2, 3]
    
        # EXPECTED BEHAVIOR
        rb = RingBuffer(3, dtype=int)  # --> rb = []
        rb.extend([1, 2, 3])           # --> rb = [1, 2, 3]
        rb.popleft(n=2, n_overlap=1)   # --> rb = [2, 3], res = [1, 2]
        rb.popleft(n=2, n_overlap=1)   # --> rb = [3], res = [2, 3]
        
        TODO: use case: n=2, n_overlap=2

* ``rotate(n=0)`` CW rotate the ring buffer by n elements.

    .. code-block:: python

        rb = RingBuffer(3, dtype=int)  # --> rb = []
        rb.extend([1, 2, 3])           # --> rb = [1, 2, 3]
        rb.rotate(1)                   # --> rb = [1, 2, 3, 0]
        TODO: error if self.is_full is False, next value is 0, e.g., capacity=5, n=1 

* ``rotateleft()`` CCW rotate the ring buffer by n elements.

    .. code-block:: python

        rb = RingBuffer(3, dtype=int)  # --> rb = []
        rb.extend([1, 2, 3])           # --> rb = [1, 2, 3]
        rb.rotateleft(1)               # --> rb = [2, 3]
        TODO: error if self.is_full is False, next value is 0, e.g., n=2


Properties
----------
* ``is_full``
* ``unwrap_address``
* ``current_address``
* ``dtype``
* ``shape``
* ``maxlen``

Indexing & slicing
------------------
* ``[]`` including negative indices and slicing

    .. code-block:: python

        from dvg_ringbuffer import RingBuffer

        rb = RingBuffer(4, dtype=int)  # --> rb[:] = array([], dtype=int32)
        rb.extend([1, 2, 3, 4, 5])     # --> rb[:] = array([2, 3, 4, 5])
        x = rb[0]                      # --> x = 2
        x = rb[-1]                     # --> x = 5
        x = rb[:3]                     # --> x = array([2, 3, 4])
        x = rb[np.array([0, 2, -1])]   # --> x = array([2, 4, 5])

        rb = RingBuffer(5, dtype=(int, 2))  # --> rb[:] = array([], shape=(0, 2), dtype=int32)
        rb.append([1, 2])                   # --> rb[:] = array([[1, 2]])
        rb.append([3, 4])                   # --> rb[:] = array([[1, 2], [3, 4]])
        rb.append([5, 6])                   # --> rb[:] = array([[1, 2], [3, 4], [5, 6]])
        x = rb[0]                           # --> x = array([1, 2])
        x = rb[0, :]                        # --> x = array([1, 2])
        x = rb[:, 0]                        # --> x = array([1, 3, 5])

"""
__author__ = "Elizabeth (Liz) A. O'Gorman"
__authoremail__ = "elizabeth.anne.ogorman@gmail.com <_> elizabeth.ogorman@duke.edu"
__url__ = "https://github.com/eaogorman/python-dvg-ringbuffer"
__date__ = "22-07-2021"
__version__ = "1.0.3" "-> 1.0.0?"

from collections.abc import Sequence
import numpy as np


# DEV NOTE:
# One might be tempted to use a `numba.njit(nogil=True)` decorator on the
# `numpy.concatenate()` functions appearing in this module, but timeit tests
# reveal that it actually hurts the performance. Numpy has already optimised its
# `concatenate()` method to maximum performance.


class RingBuffer(Sequence):
    """Manages a ring buffer with the given capacity and element type.

    Args:
        capacity (int):
            The maximum capacity of the ring buffer

        dtype (data-type, optional):
            Desired type of buffer elements. Use a type like (float, 2) to
            produce a buffer with shape (capacity, 2).

            Default: float

        allow_overwrite (bool, optional):
            If False, throw an IndexError when trying to append to an
            already full buffer.

            Default: True
    """

    def __init__(self, capacity, slide=False, s=None, dtype=float, allow_overwrite=True):
        if dtype == float:
            p = {"shape": capacity, "fill_value": np.nan, "order": "C"}
            self._arr = np.full(**p)
            self._unwrap_buffer = np.full(**p)  # @ fixed memory address
        else:
            p = {"shape": capacity, "dtype": dtype, "order": "C"}
            self._arr = np.zeros(**p)
            self._unwrap_buffer = np.zeros(**p)  # @ fixed memory address

        self._N = capacity
        self._allow_overwrite = allow_overwrite
        self._idx_L = 0  # left index
        self._idx_R = 0  # right index
        self._unwrap_buffer_is_dirty = False

    # --------------------------------------------------------------------------
    #   clear
    # --------------------------------------------------------------------------

    def clear(self):
        self._idx_L = 0
        self._idx_R = 0

        if self._arr.dtype == float:
            self._arr.fill(np.nan)
            self._unwrap_buffer.fill(np.nan)
        else:
            self._arr.fill(0)
            self._unwrap_buffer.fill(0)

    # --------------------------------------------------------------------------
    #   append
    # --------------------------------------------------------------------------

    def append(self, value):
        """Append a single value to the ring buffer.

        rb = RingBuffer(3, dtype=int)  # --> rb = []
        rb.append(1)                   # --> rb = [1]
        rb.append(2)                   # --> rb = [1, 2]
        rb.append(3)                   # --> rb = [1, 2, 3]
        rb.append(4)                   # --> rb = [2, 3, 4]
        """
        if self.is_full:
            if not self._allow_overwrite:
                raise IndexError(
                    "Append to a full RingBuffer with overwrite disabled."
                )
            if self._N == 0:
                return  # Mimick behavior of deque(maxlen=0)
            self._idx_L += 1

        self._unwrap_buffer_is_dirty = True
        self._arr[self._idx_R % self._N] = value
        self._idx_R += 1
        self._fix_indices()

    def appendleft(self, value):
        """Append a single value to the ring buffer from the left side.

        rb = RingBuffer(3, dtype=int)  # --> rb = []
        rb.appendleft(1)               # --> rb = [1]
        rb.appendleft(2)               # --> rb = [2, 1]
        rb.appendleft(3)               # --> rb = [3, 2, 1]
        rb.appendleft(4)               # --> rb = [4, 3, 2]
        """
        if self.is_full:
            if not self._allow_overwrite:
                raise IndexError(
                    "Append to a full RingBuffer with overwrite disabled."
                )
            if self._N == 0:
                return  # Mimick behavior of deque(maxlen=0)
            self._idx_R -= 1

        self._unwrap_buffer_is_dirty = True
        self._idx_L -= 1
        self._fix_indices()
        self._arr[self._idx_L] = value

    # --------------------------------------------------------------------------
    #   extend
    # --------------------------------------------------------------------------

    def extend(self, values):
        """Extend the ring buffer with a list of values.

        rb = RingBuffer(3, dtype=int)  # --> rb = []
        rb.extend([1])                 # --> rb = [1]
        rb.extend([2, 3])              # --> rb = [1, 2, 3]
        rb.extend([4, 5, 6, 7])        # --> rb = [5, 6, 7]
        """
        lv = len(values)
        if len(self) + lv > self._N:
            if not self._allow_overwrite:
                raise IndexError(
                    "RingBuffer overflows, because overwrite is disabled."
                )
            if self._N == 0:
                return  # Mimick behavior of deque(maxlen=0)

        self._unwrap_buffer_is_dirty = True
        if lv >= self._N:
            self._arr[...] = values[-self._N :]
            self._idx_R = self._N
            self._idx_L = 0
            return

        ri = self._idx_R % self._N
        sl1 = np.s_[ri : min(ri + lv, self._N)]
        sl2 = np.s_[: max(ri + lv - self._N, 0)]
        # fmt: off
        self._arr[sl1] = values[: sl1.stop - sl1.start]  # pylint: disable=no-member
        self._arr[sl2] = values[sl1.stop - sl1.start :]  # pylint: disable=no-member
        # fmt: on
        self._idx_R += lv
        self._idx_L = max(self._idx_L, self._idx_R - self._N)
        self._fix_indices()

    def extendleft(self, values):
        """Extend the ring buffer with a list of values from the left side.

        rb = RingBuffer(3, dtype=int)  # --> rb = []
        rb.extendleft([1])             # --> rb = [1]
        rb.extendleft([3, 2])          # --> rb = [3, 2, 1]
        rb.extendleft([7, 6, 5, 4])    # --> rb = [7, 6, 5]
        """
        lv = len(values)
        if len(self) + lv > self._N:
            if not self._allow_overwrite:
                raise IndexError(
                    "RingBuffer overflows, because overwrite is disabled."
                )
            if self._N == 0:
                return  # Mimick behavior of deque(maxlen=0)

        self._unwrap_buffer_is_dirty = True
        if lv >= self._N:
            self._arr[...] = values[: self._N]
            self._idx_R = self._N
            self._idx_L = 0
            return

        self._idx_L -= lv
        self._fix_indices()
        li = self._idx_L
        sl1 = np.s_[li : min(li + lv, self._N)]
        sl2 = np.s_[: max(li + lv - self._N, 0)]
        # fmt: off
        self._arr[sl1] = values[:sl1.stop - sl1.start]  # pylint: disable=no-member
        self._arr[sl2] = values[sl1.stop - sl1.start:]  # pylint: disable=no-member
        # fmt: on
        self._idx_R = min(self._idx_R, self._idx_L + self._N)

    # --------------------------------------------------------------------------
    #   pop
    # --------------------------------------------------------------------------
    
    def pop(self, n=1, n_overlap=0):
        """Remove n element(s) from the right side of the ring buffer and return it. Include n_overlap in remaining buffer.

        rb = RingBuffer(3, dtype=int)  # --> rb = []
        rb.extend([1, 2, 3])           # --> rb = [1, 2, 3]
        res = rb.pop()                       # --> rb = [1, 2], res = [3]
        res = rb.pop(n=2)                      # --> rb = [], res = [1, 2]

        # EXPECTED BEHAVIOR
        rb = RingBuffer(3, dtype=int)  # --> rb = []
        rb.extend([1, 2, 3])           # --> rb = [1, 2, 3]
        res = rb.pop(n=2, n_overlap=1)       # --> rb = [1, 2], res = [2, 3]
        res = rb.pop(n=2, n_overlap=1)       # --> rb = [1], res = [1, 2]
        
        TODO: use case: n=2, n_overlap=2
        """
        if len(self) == 0:
            raise IndexError("Pop from an empty RingBuffer.")
        self._unwrap_buffer_is_dirty = True
        self._idx_R -= n
        if n > 1:
            res = self._arr[(self._idx_R % self._N):(self._idx_R % self._N + n)]
        else:
            res = self._arr[(self._idx_R % self._N)]
        if n_overlap > 0:
            self._idx_R += n_overlap
        self._fix_indices()
        return res

    def popleft(self, n=1, n_overlap=0):
        """Remove n element(s) from the left side of the ring buffer and return it. Include n_overlap in remaining buffer.

        rb = RingBuffer(3, dtype=int)  # --> rb = []
        rb.extend([1, 2, 3])           # --> rb = [1, 2, 3]
        res = rb.popleft()             # --> rb = [2, 3], res = [1]
        res = rb.popleft(n=2)            # --> rb = [], res = [2, 3]
    
        # EXPECTED BEHAVIOR
        rb = RingBuffer(3, dtype=int)  # --> rb = []
        rb.extend([1, 2, 3])           # --> rb = [1, 2, 3]
        rb.popleft(n=2, n_overlap=1)   # --> rb = [2, 3], res = [1, 2]
        rb.popleft(n=2, n_overlap=1)   # --> rb = [3], res = [2, 3]
        
        TODO: use case: n=2, n_overlap=2
        """
        if len(self) == 0:
            raise IndexError("Pop from an empty RingBuffer.")
        self._unwrap_buffer_is_dirty = True
        if n > 1:
            res = self._arr[self._idx_L:(self._idx_L + n)]
        else: 
            res = self._arr[self._idx_L]
        if n_overlap > 0:
            self._idx_L += (n - n_overlap)
        else: 
            self._idx_L += n
        self._fix_indices()
        return res
    
    # --------------------------------------------------------------------------
    #   rotate
    # --------------------------------------------------------------------------

    # NOTE: HAS BUGS
    # TODO: FIX — TEST EDGE CASES
    def rotate(self, n=0):      
        """CW rotate the ring buffer by n elements.

        rb = RingBuffer(3, dtype=int)  # --> rb = []
        rb.extend([1, 2, 3])           # --> rb = [1, 2, 3]
        rb.rotate(1)                   # --> rb = [1, 2, 3, 0]
        TODO: error if self.is_full is False, next value is 0, e.g., capacity=5, n=1 
        """
        self._idx_R += n

    def rotateleft(self, n=0):
        """CCW rotate left to an arbitrary value in the ring buffer.

        rb = RingBuffer(3, dtype=int)  # --> rb = []
        rb.extend([1, 2, 3])           # --> rb = [1, 2, 3]
        rb.rotateleft(1)               # --> rb = [2, 3]
        TODO: error if self.is_full is False, next value is 0, e.g., n=2
        """
        self._idx_L += n

    # --------------------------------------------------------------------------
    #   Properties
    # --------------------------------------------------------------------------

    @property
    def is_full(self):
        return len(self) == self._N

    @property
    def unwrap_address(self):
        """Get the fixed memory address of the internal unwrap buffer, used when
        the ring buffer is completely full.
        """
        return self._unwrap_buffer[:].__array_interface__["data"][0]

    @property
    def current_address(self):
        """Get the current memory address of the array behind the buffer.
        """
        return self[:].__array_interface__["data"][0]

    @property
    def dtype(self):
        return self._arr.dtype

    @property
    def shape(self):
        return (len(self),) + self._arr.shape[1:]

    @property
    def maxlen(self):
        return self._N

    # --------------------------------------------------------------------------
    #   _unwrap
    # --------------------------------------------------------------------------

    def _unwrap(self):
        """Copy the data from this buffer into unwrapped form.
        """
        return np.concatenate(
            (
                self._arr[self._idx_L : min(self._idx_R, self._N)],
                self._arr[: max(self._idx_R - self._N, 0)],
            )
        )

    # --------------------------------------------------------------------------
    #   _unwrap_into_buffer
    # --------------------------------------------------------------------------

    def _unwrap_into_buffer(self):
        """Copy the data from this buffer into unwrapped form to the unwrap
        buffer at a fixed memory address. Only call when the buffer is full.
        """
        if self._unwrap_buffer_is_dirty:
            # print("Unwrap buffer was dirty")
            np.concatenate(
                (
                    self._arr[self._idx_L : min(self._idx_R, self._N)],
                    self._arr[: max(self._idx_R - self._N, 0)],
                ),
                out=self._unwrap_buffer,
            )
            self._unwrap_buffer_is_dirty = False
        else:
            # print("Unwrap buffer was clean")
            pass

    # --------------------------------------------------------------------------
    #   _fix_indices
    # --------------------------------------------------------------------------

    def _fix_indices(self):
        """Enforce our invariant that 0 <= self._idx_L < self._N.
        """
        if self._idx_L >= self._N:
            self._idx_L -= self._N
            self._idx_R -= self._N
        elif self._idx_L < 0:
            self._idx_L += self._N
            self._idx_R += self._N

    # --------------------------------------------------------------------------
    #   Dunder methods
    # --------------------------------------------------------------------------

    def __array__(self):
        """Numpy compatibility
        """
        # print("__array__")
        if self.is_full:
            self._unwrap_into_buffer()
            return self._unwrap_buffer
        else:
            return self._unwrap()

    def __len__(self):
        return self._idx_R - self._idx_L

    def __getitem__(self, item):

        # --------------------------
        #   ringbuffer[slice]
        #   ringbuffer[tuple]
        #   ringbuffer[None]
        # --------------------------

        if isinstance(item, (slice, tuple)) or item is None:
            if self.is_full:
                # print("  --> _unwrap_buffer[item]")
                self._unwrap_into_buffer()
                return self._unwrap_buffer[item]

            # print("  --> _unwrap()[item]")
            return self._unwrap()[item]

        # ----------------------------------
        #   ringbuffer[int]
        #   ringbuffer[list of ints]
        #   ringbuffer[np.ndarray of ints]
        # ----------------------------------
        item_arr = np.asarray(item)

        if not issubclass(item_arr.dtype.type, np.integer):
            raise TypeError("RingBuffer indices must be integers.")

        if len(self) == 0:
            raise IndexError(
                "RingBuffer list index out of range. The RingBuffer has "
                "length 0."
            )

        if not hasattr(item, "__len__"):
            # Single element: We can speed up the code
            # print("  --> single element")

            # Check for `List index out of range`
            if item_arr < -len(self) or item_arr >= len(self):
                raise IndexError(
                    "RingBuffer list index %s out of range. The RingBuffer "
                    "has length %s." % (item_arr, len(self))
                )

            if item_arr < 0:
                item_arr = (self._idx_R + item_arr) % self._N
            else:
                item_arr = (item_arr + self._idx_L) % self._N

        else:
            # Multiple elements
            # print("  --> multiple elements")

            # Check for `List index out of range`
            if np.any(item_arr < -len(self)) or np.any(item_arr >= len(self)):
                idx_under = item_arr[np.where(item_arr < -len(self))]
                idx_over = item_arr[np.where(item_arr >= len(self))]
                idx_oor = np.sort(np.concatenate((idx_under, idx_over)))
                raise IndexError(
                    "RingBuffer list indices %s out of range. The RingBuffer "
                    "has length %s." % (idx_oor, len(self))
                )
            idx_neg = np.where(item_arr < 0)
            idx_pos = np.where(item_arr >= 0)

            if len(idx_neg) > 0:
                item_arr[idx_neg] = (self._idx_R + item_arr[idx_neg]) % self._N
            if len(idx_pos) > 0:
                item_arr[idx_pos] = (item_arr[idx_pos] + self._idx_L) % self._N

        # print("  --> _arr[item_arr]")
        return self._arr[item_arr]

    def __iter__(self):
        # print("__iter__")
        if self.is_full:
            self._unwrap_into_buffer()
            return iter(self._unwrap_buffer)
        else:
            return iter(self._unwrap())

    def __repr__(self):
        return "<RingBuffer of {!r}>".format(np.asarray(self))