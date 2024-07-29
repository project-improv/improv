import os
import uuid

import pickle
import logging
import traceback

import numpy as np
import pyarrow.plasma as plasma

from redis import Redis
from redis.retry import Retry
from redis.backoff import ConstantBackoff
from redis.exceptions import BusyLoadingError, ConnectionError, TimeoutError

from scipy.sparse import csc_matrix
from pyarrow.lib import ArrowIOError
from pyarrow._plasma import PlasmaObjectExists, ObjectNotAvailable

REDIS_GLOBAL_TOPIC = "global_topic"

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class StoreInterface:
    """General interface for a store"""

    def get(self):
        raise NotImplementedError

    def put(self):
        raise NotImplementedError

    def delete(self):
        raise NotImplementedError

    def replace(self):
        raise NotImplementedError

    def subscribe(self):
        raise NotImplementedError


class RedisStoreInterface(StoreInterface):
    def __init__(self, name="default", server_port_num=6379, hostname="localhost"):
        self.name = name
        self.server_port_num = server_port_num
        self.hostname = hostname
        self.client = self.connect_to_server()

    def connect_to_server(self):
        # TODO this should scan for available ports, but only if configured to do so.
        # This happens when the config doesn't have Redis settings,
        # so we need to communicate this somehow to the StoreInterface here.
        """Connect to the store at store_loc, max 20 retries to connect
        Raises exception if can't connect
        Returns the Redis client if successful

        Args:
            server_port_num: the port number where the Redis server
            is running on localhost.
        """
        try:
            retry = Retry(ConstantBackoff(0.25), 5)
            self.client = Redis(
                host=self.hostname,
                port=self.server_port_num,
                retry=retry,
                retry_on_timeout=True,
                retry_on_error=[
                    BusyLoadingError,
                    ConnectionError,
                    TimeoutError,
                    ConnectionRefusedError,
                ],
            )
            self.client.ping()
            logger.info(
                "Successfully connected to redis datastore on port {} ".format(
                    self.server_port_num
                )
            )
        except Exception:
            logger.exception(
                "Cannot connect to redis datastore on port {}".format(
                    self.server_port_num
                )
            )
            raise CannotConnectToStoreInterfaceError(self.server_port_num)

        return self.client

    def put(self, object):
        """
        Put a single object referenced by its string name
        into the store. If the store already has a value stored at this key,
        the value will not be overwritten.

        Unknown error

        Args:
            object: the object to store in Redis
            object_key (str): the key under which the object should be stored

        Returns:
            object: the object that was a
        """
        object_key = str(os.getpid()) + str(uuid.uuid4())
        try:
            # buffers would theoretically go here if we need to force out-of-band
            # serialization for single objects
            # TODO this will actually just silently fail if we use an existing
            # TODO key; not sure it's worth the network overhead to check every
            # TODO key twice every time. we still need a better solution for
            # TODO this, but it will work now singlethreaded most of the time.

            self.client.set(object_key, pickle.dumps(object, protocol=5), nx=True)
        except Exception:
            logger.error("Could not store object {}".format(object_key))
            logger.error(traceback.format_exc())

        return object_key

    def get(self, object_key):
        """
        Get object by specified key

        Args:
            object_name: the key of the object

        Returns:
            Stored object

        Raises:
            ObjectNotFoundError: If the key is not found
        """
        object_value = self.client.get(object_key)
        if object_value:
            # buffers would also go here to force out-of-band deserialization
            return pickle.loads(object_value)

        logger.warning("Object {} cannot be found.".format(object_key))
        raise ObjectNotFoundError

    def subscribe(self, topic=REDIS_GLOBAL_TOPIC):
        p = self.client.pubsub()
        p.subscribe(topic)

    def get_list(self, ids):
        """Get multiple objects from the store

        Args:
            ids (list): of type str

        Returns:
            list of the objects
        """
        return self.client.mget(ids)

    def get_all(self):
        """Get a listing of all objects in the store.
        Note that this may be very performance-intensive in large databases.

        Returns:
            list of all the objects in the store
        """
        all_keys = self.client.keys()  # defaults to "*" pattern, so will fetch all
        return self.client.mget(all_keys)

    def reset(self):
        """Reset client connection"""
        self.client = self.connect_to_server()
        logger.debug(
            "Reset local connection to store on port: {0}".format(self.server_port_num)
        )

    def notify(self):
        pass  # I don't see any call sites for this, so leaving it blank at the moment


class PlasmaStoreInterface(StoreInterface):
    """Basic interface for our specific data store implemented with apache arrow plasma
    Objects are stored with object_ids
    References to objects are contained in a dict where key is shortname,
    value is object_id
    """

    def __init__(self, name="default", store_loc="/tmp/store"):
        """
        Constructor for the StoreInterface

        :param name:
        :param store_loc: Apache Arrow Plasma client location
        """

        self.name = name
        self.store_loc = store_loc
        self.client = self.connect_store(store_loc)
        self.stored = {}

    def connect_store(self, store_loc):
        """Connect to the store at store_loc, max 20 retries to connect
        Raises exception if can't connect
        Returns the plasmaclient if successful
        Updates the client internal

        Args:
            store_loc: store location
        """
        try:
            self.client = plasma.connect(store_loc, 20)
            logger.info("Successfully connected to store: {} ".format(store_loc))
        except Exception:
            logger.exception("Cannot connect to store: {}".format(store_loc))
            raise CannotConnectToStoreInterfaceError(store_loc)
        return self.client

    def put(self, object, object_name):
        """
        Put a single object referenced by its string name
        into the store
        Raises PlasmaObjectExists if we are overwriting
        Unknown error

        Args:
            object:
            object_name (str):
            flush_this_immediately (bool):

        Returns:
            class 'plasma.ObjectID': Plasma object ID

        Raises:
            PlasmaObjectExists: if we are overwriting \
            unknown error
        """
        object_id = None
        try:
            # Need to pickle if object is csc_matrix
            if isinstance(object, csc_matrix):
                prot = pickle.HIGHEST_PROTOCOL
                object_id = self.client.put(pickle.dumps(object, protocol=prot))
            else:
                object_id = self.client.put(object)

        except PlasmaObjectExists:
            logger.error("Object already exists. Meant to call replace?")
        except ArrowIOError:
            logger.error("Could not store object {}".format(object_name))
            logger.info("Refreshing connection and continuing")
            self.reset()
        except Exception:
            logger.error("Could not store object {}".format(object_name))
            logger.error(traceback.format_exc())

        return object_id

    def get(self, object_name):
        """Get a single object from the store by object name
        Checks to see if it knows the object first
        Otherwise throw CannotGetObject to request dict update
        TODO: update for lists of objects
        TODO: replace with getID

        Returns:
            Stored object
        """
        # print('trying to get ', object_name)
        # if self.stored.get(object_name) is None:
        #     logger.error('Never recorded storing this object: '+object_name)
        #     # Don't know anything about this object, treat as problematic
        #     raise CannotGetObjectError(query = object_name)
        # else:
        return self.getID(object_name)

    def getID(self, obj_id):
        """
        Get object by object ID

        Args:
            obj_id (class 'plasma.ObjectID'): the id of the object

        Returns:
            Stored object

        Raises:
            ObjectNotFoundError: If the id is not found
        """
        res = self.client.get(obj_id, 0)  # Timeout = 0 ms
        if res is not plasma.ObjectNotAvailable:
            return res if not isinstance(res, bytes) else pickle.loads(res)

        logger.warning("Object {} cannot be found.".format(obj_id))
        raise ObjectNotFoundError

    def getList(self, ids):
        """Get multiple objects from the store

        Args:
            ids (list): of type plasma.ObjectID

        Returns:
            list of the objects
        """
        # self._get()
        return self.client.get(ids)

    def get_all(self):
        """Get a listing of all objects in the store

        Returns:
            list of all the objects in the store
        """
        return self.client.list()

    def reset(self):
        """Reset client connection"""
        self.client = self.connect_store(self.store_loc)
        logger.debug("Reset local connection to store: {0}".format(self.store_loc))

    def release(self):
        self.client.disconnect()

    # Subscribe to notifications about sealed objects?
    def subscribe(self):
        """Subscribe to a section? of the ds for singals

        Raises:
            Exception: Unknown error
        """
        try:
            self.client.subscribe()
        except Exception as e:
            logger.error("Unknown error: {}".format(e))
            raise Exception

    def notify(self):
        try:
            notification_info = self.client.get_next_notification()
            # recv_objid, recv_dsize, recv_msize = notification_info
        except ArrowIOError:
            notification_info = None
        except Exception as e:
            logger.exception("Notification error: {}".format(e))
            raise Exception

        return notification_info

    # Necessary? plasma.ObjectID.from_random()
    def random_ObjectID(self, number=1):
        ids = []
        for i in range(number):
            ids.append(plasma.ObjectID(np.random.bytes(20)))
        return ids

    def updateStoreInterfaced(self, object_name, object_id):
        """Update local dict with info we need locally
        Report to Nexus that we updated the store
        (did a put or delete/replace)

        Args:
            object_name (str): the name of the object to update
            object_id (): the id of the object to update
        """
        self.stored.update({object_name: object_id})

    def getStored(self):
        """
        Returns:
            its info about what it has stored
        """
        return self.stored

    def _put(self, obj, id):
        """Internal put"""
        return self.client.put(obj, id)

    def _get(self, object_name):
        """Get an object from the store using its name
        Assumes we know the id for the object_name

        Raises:
            ObjectNotFound: if object_id returns no object from the store
        """
        # Most errors not shown to user.
        # Maintain separation between external and internal function calls.
        res = self.getID(self.stored.get(object_name))
        # Can also use contains() to check

        logger.warning("{}".format(object_name))
        if isinstance(res, ObjectNotAvailable):
            logger.warning("Object {} cannot be found.".format(object_name))
            raise ObjectNotFoundError(obj_id_or_name=object_name)  # TODO: Don't raise?
        else:
            return res


StoreInterface = RedisStoreInterface


class ObjectNotFoundError(Exception):
    def __init__(self, obj_id_or_name):
        super().__init__()

        self.name = "ObjectNotFoundError"
        self.obj_id_or_name = obj_id_or_name

        # TODO: self.message does not properly receive obj_id_or_name
        self.message = 'Cannnot find object with ID/name "{}"'.format(obj_id_or_name)

    def __str__(self):
        return self.message


class CannotGetObjectError(Exception):
    def __init__(self, query):
        super().__init__()

        self.name = "CannotGetObjectError"
        self.query = query
        self.message = "Cannot get object {}".format(self.query)

    def __str__(self):
        return self.message


class CannotConnectToStoreInterfaceError(Exception):
    """Raised when failing to connect to store."""

    def __init__(self, store_loc):
        super().__init__()

        self.name = "CannotConnectToStoreInterfaceError"

        self.message = "Cannot connect to store at {}".format(str(store_loc))

    def __str__(self):
        return self.message
