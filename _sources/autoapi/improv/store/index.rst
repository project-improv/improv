:py:mod:`improv.store`
======================

.. py:module:: improv.store


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   improv.store.StoreInterface
   improv.store.PlasmaStoreInterface
   improv.store.LMDBStoreInterface
   improv.store.LMDBData




Attributes
~~~~~~~~~~

.. autoapisummary::

   improv.store.logger
   improv.store.StoreInterface


.. py:data:: logger

   

.. py:class:: StoreInterface

   General interface for a store

   .. py:method:: get()
      :abstractmethod:


   .. py:method:: put()
      :abstractmethod:


   .. py:method:: delete()
      :abstractmethod:


   .. py:method:: replace()
      :abstractmethod:


   .. py:method:: subscribe()
      :abstractmethod:



.. py:class:: PlasmaStoreInterface(name='default', store_loc='/tmp/store')

   Bases: :py:obj:`StoreInterface`

   Basic interface for our specific data store implemented with apache arrow plasma
   Objects are stored with object_ids
   References to objects are contained in a dict where key is shortname,
   value is object_id

   .. py:method:: connect_store(store_loc)

      Connect to the store at store_loc, max 20 retries to connect
      Raises exception if can't connect
      Returns the plasmaclient if successful
      Updates the client internal

      :param store_loc: store location


   .. py:method:: put(object, object_name)

      Put a single object referenced by its string name
      into the store
      Raises PlasmaObjectExists if we are overwriting
      Unknown error

      :param object:
      :param object_name:
      :type object_name: str
      :param flush_this_immediately:
      :type flush_this_immediately: bool

      :returns: Plasma object ID
      :rtype: class 'plasma.ObjectID'

      :raises PlasmaObjectExists: if we are overwriting             unknown error


   .. py:method:: get(object_name)

      Get a single object from the store by object name
      Checks to see if it knows the object first
      Otherwise throw CannotGetObject to request dict update
      TODO: update for lists of objects
      TODO: replace with getID

      :returns: Stored object


   .. py:method:: getID(obj_id, hdd_only=False)

      Get object by object ID

      :param obj_id: the id of the object
      :type obj_id: class 'plasma.ObjectID'
      :param hdd_only:
      :type hdd_only: bool

      :returns: Stored object

      :raises ObjectNotFoundError: If the id is not found


   .. py:method:: getList(ids)

      Get multiple objects from the store

      :param ids: of type plasma.ObjectID
      :type ids: list

      :returns: list of the objects


   .. py:method:: get_all()

      Get a listing of all objects in the store

      :returns: list of all the objects in the store


   .. py:method:: reset()

      Reset client connection


   .. py:method:: release()


   .. py:method:: subscribe()

      Subscribe to a section? of the ds for singals

      :raises Exception: Unknown error


   .. py:method:: notify()


   .. py:method:: random_ObjectID(number=1)


   .. py:method:: updateStoreInterfaced(object_name, object_id)

      Update local dict with info we need locally
      Report to Nexus that we updated the store
      (did a put or delete/replace)

      :param object_name: the name of the object to update
      :type object_name: str
      :param object_id (): the id of the object to update


   .. py:method:: getStored()

      :returns: its info about what it has stored



.. py:class:: LMDBStoreInterface(path='../outputs/', name=None, load=False, max_size=1000000000000.0, flush_immediately=False, commit_freq=1)

   Bases: :py:obj:`StoreInterface`

   General interface for a store

   .. py:method:: get(key: Union[pyarrow.plasma.ObjectID, bytes, List[pyarrow.plasma.ObjectID], List[bytes]], include_metadata=False)

      Get object using key (could be any byte string or plasma.ObjectID)

      :param key:
      :param include_metadata: returns whole LMDBData if true else LMDBData.obj
                               (just the stored object).
      :type include_metadata: bool

      :returns: object or LMDBData


   .. py:method:: get_keys()

      Get all keys in LMDB


   .. py:method:: put(obj, obj_name, obj_id=None, flush_this_immediately=False)

      Put object ID / object pair into LMDB.

      :param obj: Object to be saved
      :param obj_name: the name of the object
      :type obj_name: str
      :param obj_id: Object_id from Plasma client
      :type obj_id: 'plasma.ObjectID'
      :param flush_this_immediately: Override self.flush_immediately.
                                     For storage of critical objects.
      :type flush_this_immediately: bool

      :returns: None


   .. py:method:: flush(sig=None, frame=None)

      Must run before exiting. Flushes buffer to disk.


   .. py:method:: commit_daemon()


   .. py:method:: commit()

      Commit objects in {self.put_cache} into LMDB.


   .. py:method:: delete(obj_id)

      Delete object from LMDB.

      :param obj_id: the object_id to be deleted
      :type obj_id: class 'plasma.ObjectID'

      :returns: None

      :raises ObjectNotFoundError: If the id is not found


   .. py:method:: replace()


   .. py:method:: subscribe()



.. py:data:: StoreInterface

   

.. py:class:: LMDBData

   Dataclass to store objects and their metadata into LMDB.

   .. py:property:: queue

      Returns:
      Queue name if object is a queue else None

   .. py:attribute:: obj
      :type: object

      

   .. py:attribute:: time
      :type: float

      

   .. py:attribute:: name
      :type: str

      

   .. py:attribute:: is_queue
      :type: bool
      :value: False

      


.. py:exception:: ObjectNotFoundError(obj_id_or_name)

   Bases: :py:obj:`Exception`

   Common base class for all non-exit exceptions.


.. py:exception:: CannotGetObjectError(query)

   Bases: :py:obj:`Exception`

   Common base class for all non-exit exceptions.


.. py:exception:: CannotConnectToStoreInterfaceError(store_loc)

   Bases: :py:obj:`Exception`

   Raised when failing to connect to store.


