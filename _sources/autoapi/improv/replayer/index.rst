:py:mod:`improv.replayer`
=========================

.. py:module:: improv.replayer


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   improv.replayer.Replayer




.. py:class:: Replayer(*args, lmdb_path, replay: str, resave=False, **kwargs)

   Bases: :py:obj:`nexus.actor.Actor`

   .. py:method:: get_lmdb_values(replay: str, func: Callable = None) -> List[nexus.store.LMDBData]

      Load saved queue objects from LMDB

      :param replay: named of Actor
      :param func: (optional) Function to apply to objects before returning

      :returns: lmdb_values


   .. py:method:: setup()


   .. py:method:: move_to_plasma(lmdb_values)

      Put objects into current plasma store and update object ID in saved queue.

      :param lmdb_values:


   .. py:method:: put_setup(lmdb_values)

      Put all objects created before Run into queue immediately.

      :param lmdb_values:


   .. py:method:: run()


   .. py:method:: runner()

      Get list of objects and output them
      to their respective queues based on time delay.



