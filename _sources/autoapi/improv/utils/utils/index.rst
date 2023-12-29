:py:mod:`improv.utils.utils`
============================

.. py:module:: improv.utils.utils


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   improv.utils.utils.get_num_length_from_key



.. py:function:: get_num_length_from_key()

   Coroutine that gets the length of digits in LMDB key.
   Assumes that object name does not have any digits.

   For example:
       FileAcquirer puts objects with names 'acq_raw{i}' where i is the frame number.
       {i}, however, is not padded with zero, so the length changes with number.
       The B-tree sorting in LMDB results in messed up number sorting.
   .. rubric:: Example

   >>> num_idx = get_num_length_from_key()
   >>> num_idx.send(b'acq_raw1GA×L°°.')
   1


