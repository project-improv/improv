:py:mod:`improv.utils.reader`
=============================

.. py:module:: improv.utils.reader


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   improv.utils.reader.LMDBReader




.. py:class:: LMDBReader(path)

   .. py:method:: get_all_data()

      Load all data from LMDB into a dictionary
      Make sure that the LMDB is small enough to fit in RAM


   .. py:method:: get_data_types()

      Return all data types defined as {object_name}, but without number.


   .. py:method:: get_data_by_number(t)

      Return data at a specific frame number t


   .. py:method:: get_data_by_type(t)

      Return data with key that starts with t


   .. py:method:: get_params()

      Return parameters in a dictionary



