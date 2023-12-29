:py:mod:`improv.link`
=====================

.. py:module:: improv.link


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   improv.link.AsyncQueue
   improv.link.MultiAsyncQueue



Functions
~~~~~~~~~

.. autoapisummary::

   improv.link.Link
   improv.link.MultiLink



Attributes
~~~~~~~~~~

.. autoapisummary::

   improv.link.logger


.. py:data:: logger

   

.. py:function:: Link(name, start, end)

   Function to construct a queue that Nexus uses for
   inter-process (actor) signaling and information passing.

   A Link has an internal queue that can be synchronous (put, get)
   as inherited from multiprocessing.Manager.Queue
   or asynchronous (put_async, get_async) using async executors.

   :param See AsyncQueue constructor:

   :returns: queue for communicating between actors and with Nexus
   :rtype: AsyncQueue


.. py:class:: AsyncQueue(q, name, start, end)

   Bases: :py:obj:`object`

   Single-output and asynchronous queue class.

   .. attribute:: queue

      

   .. attribute:: real_executor

      

   .. attribute:: cancelled_join

      boolean

   .. attribute:: name

      

   .. attribute:: start

      

   .. attribute:: end

      

   .. attribute:: status

      

   .. attribute:: result

      

   .. attribute:: num

      

   .. attribute:: dict

      

   .. py:method:: getStart()

      Gets the starting actor.

      The starting actor is the actor that is at the tail of the link.
      This actor is the one that gives output.

      :returns: The starting actor name
      :rtype: start (str)


   .. py:method:: getEnd()

      Gets the ending actor.

      The ending actor is the actor that is at the head of the link.
      This actor is the one that takes input.

      :returns: The ending actor name
      :rtype: end (str)


   .. py:method:: put(item)

      Function wrapper for put.

      :param item: Any item that can be sent through a queue
      :type item: object


   .. py:method:: put_nowait(item)

      Function wrapper for put without waiting

      :param item: Any item that can be sent through a queue
      :type item: object


   .. py:method:: put_async(item)
      :async:

      Coroutine for an asynchronous put

      It adds the put request to the event loop and awaits.

      :param item: Any item that can be sent through a queue
      :type item: object

      :returns: Awaitable or result of the put


   .. py:method:: get_async()
      :async:

      Coroutine for an asynchronous get

      It adds the get request to the event loop and awaits, setting
      the status to pending. Once the get has returned, it returns the
      result of the get and sets its status as done.

      Explicitly passes any exceptions to not hinder execution.
      Errors are logged with the get_async tag.

      :returns: Awaitable or result of the get.

      :raises CancelledError: task is cancelled
      :raises EOFError:
      :raises FileNotFoundError:
      :raises Exception:


   .. py:method:: cancel_join_thread()

      Function wrapper for cancel_join_thread.


   .. py:method:: join_thread()

      Function wrapper for join_thread.



.. py:function:: MultiLink(name, start, end)

   Function to generate links for the multi-output queue case.

   :param See constructor for AsyncQueue or MultiAsyncQueue:

   :returns: Producer end of the queue
             List: AsyncQueues for consumers
   :rtype: MultiAsyncQueue


.. py:class:: MultiAsyncQueue(q_in, q_out, name, start, end)

   Bases: :py:obj:`AsyncQueue`

   Extension of AsyncQueue to have multiple endpoints.

   Inherits from AsyncQueue.
   A single producer queue's 'put' is copied to multiple consumer's
   queues, q_in is the producer queue, q_out are the consumer queues.

   .. todo:: Test the async nature of this group of queues

   .. py:method:: put(item)

      Function wrapper for put.

      :param item: Any item that can be sent through a queue
      :type item: object


   .. py:method:: put_nowait(item)

      Function wrapper for put without waiting

      :param item: Any item that can be sent through a queue
      :type item: object



