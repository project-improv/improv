import os
import uuid

import pickle
import logging
import zlib

from redis import Redis
from redis.retry import Retry
from redis.backoff import ConstantBackoff
from redis.exceptions import BusyLoadingError, ConnectionError, TimeoutError

ZLIB_COMPRESSION_LEVEL = -1

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
    def __init__(
        self,
        name="default",
        server_port_num=6379,
        hostname="localhost",
        compression_level=ZLIB_COMPRESSION_LEVEL,
    ):
        self.name = name
        self.server_port_num = server_port_num
        self.hostname = hostname
        self.client = self.connect_to_server()
        self.compression_level = compression_level

    def connect_to_server(self):
        """Connect to the store, max 20 retries to connect
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
        data = zlib.compress(
            pickle.dumps(object, protocol=5), level=self.compression_level
        )

        self.client.set(object_key, data, nx=True)

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
            return pickle.loads(zlib.decompress(object_value))

        logger.warning("Object {} cannot be found.".format(object_key))
        raise ObjectNotFoundError(object_key)

    def get_list(self, ids):
        """Get multiple objects from the store

        Args:
            ids (list): of type str

        Returns:
            list of the objects
        """
        return [
            pickle.loads(zlib.decompress(object_value))
            for object_value in self.client.mget(ids)
        ]

    def get_all(self):
        """Get a listing of all objects in the store.
        Note that this may be very performance-intensive in large databases.

        Returns:
            list of all the objects in the store
        """
        all_keys = self.client.keys()  # defaults to "*" pattern, so will fetch all
        return self.get_list(all_keys)

    def reset(self):
        """Reset client connection"""
        self.client = self.connect_to_server()
        logger.debug(
            "Reset local connection to store on port: {0}".format(self.server_port_num)
        )


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

    def __init__(self, store_port):
        super().__init__()

        self.name = "CannotConnectToStoreInterfaceError"

        self.message = "Cannot connect to store at {}".format(str(store_port))

    def __str__(self):
        return self.message
