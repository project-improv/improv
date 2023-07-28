import pickle
import time
import numpy as np
import pyarrow.plasma as plasma
from scipy.sparse import csc_matrix
import signal

from dataclasses import dataclass, make_dataclass
from queue import Queue
from pathlib import Path
from random import random
from threading import Thread
from typing import List, Union

import lmdb
from pyarrow.lib import ArrowIOError
from pyarrow._plasma import PlasmaObjectExists, ObjectNotAvailable, ObjectID

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# TODO: Use Apache Arrow for better memory usage with the Plasma store


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


class PlasmaStoreInterface(StoreInterface):
    """Basic interface for our specific data store implemented with apache arrow plasma
    Objects are stored with object_ids
    References to objects are contained in a dict where key is shortname,
    value is object_id
    """

    def __init__(
        self,
        name="default",
        store_loc="/tmp/store",
        use_lmdb=False,
        lmdb_path="../outputs/",
        lmdb_name=None,
        hdd_maxstore=1e12,
        flush_immediately=False,
        commit_freq=1,
    ):
        """
        Constructor for the StoreInterface

        :param name:
        :param store_loc: Apache Arrow Plasma client location

        :param use_lmdb: bool Also write data to disk using the LMDB

        :param hdd_maxstore:
            Maximum size database may grow to; used to size the memory mapping.
            If the database grows larger than map_size, a MapFullError will be raised.
            On 64-bit there is no penalty for making this huge. Must be <2GB on 32-bit.

        :param hdd_path: Path to LMDB folder.
        :param flush_immediately: Save objects to disk immediately
        :param commit_freq: If not flush_immediately,
            flush data to disk every {commit_freq} seconds.
        """

        self.name = name
        self.store_loc = store_loc
        self.client = self.connect_store(store_loc)
        self.stored = {}

        # Offline db
        self.use_hdd = use_lmdb
        self.flush_immediately = flush_immediately

        if use_lmdb:
            self.lmdb_store = LMDBStoreInterface(
                path=lmdb_path,
                name=lmdb_name,
                max_size=hdd_maxstore,
                flush_immediately=flush_immediately,
                commit_freq=commit_freq,
            )

    def connect_store(self, store_loc):
        """Connect to the store at store_loc
        Raises exception if can't connect
        Returns the plasmaclient if successful
        Updates the client internal
        """
        logger.warning("attempting to connect to store")
        try:
            self.client = plasma.connect(store_loc, 1)
            # Is plasma.PlasmaClient necessary?
            # 20 in plasma.connect(store_loc, 20) = 20 retries
            # self.client: plasma.PlasmaClient = plasma.connect(store_loc, 20)
            logger.info(
                "Successfully connected to store at locations {0} ".format(store_loc)
            )
        except Exception:
            logger.exception("Cannot connect to store: {0}".format(store_loc))
            raise CannotConnectToStoreInterfaceError(store_loc)
        return self.client

    def put(self, object, object_name, flush_this_immediately=False):
        """
        Put a single object referenced by its string name
        into the store
        Raises PlasmaObjectExists if we are overwriting
        Unknown error

        :param obj:
        :param object_name:
        :type object_name: str
        :param flush_this_immediately:
        :return: Plasma object ID
        :rtype: class 'plasma.ObjectID'
        """

        object_id = None
        try:
            # Need to pickle if object is csc_matrix
            if isinstance(object, csc_matrix):
                object_id = self.client.put(
                    pickle.dumps(object, protocol=pickle.HIGHEST_PROTOCOL)
                )
            else:
                object_id = self.client.put(object)

            if self.use_hdd:
                self.lmdb_store.put(object, object_name, obj_id=object_id)
        except PlasmaObjectExists:
            logger.error("Object already exists. Meant to call replace?")
        except ArrowIOError as e:
            logger.error(
                "Could not store object "
                + object_name
                + ": {} {}".format(type(e).__name__, e)
            )
            logger.info("Refreshing connection and continuing")
            self.reset()
        except Exception as e:
            logger.error(
                "Could not store object "
                + object_name
                + ": {} {}".format(type(e).__name__, e)
            )

        return object_id

        # object_id = None

        # try:
        #     # Need to pickle if object is csc_matrix
        #     # csc needed for CaImAn
        #     # Pyarrow for csc/other sparse arrays
        #     # All non-arrow objects must be pickle-able
        #     # Write more general try-catch,
        #     #  if anything user wants to put in store returns cannot put -
        #     #  pickle first, then store
        #     # What else could we not put in?
        #     # List of test objects that cannot be stored
        #     # https://stackoverflow.com/questions/17872056/
        #       how-to-check-if-an-object-is-pickleable
        #       #:~:text=In%20python%20you%20can%20check,(x%2C%20Number).%22
        #     try:
        #         pickle.dumps(object)
        #     except pickle.PicklingError:
        #         return False
        #     return True
        #     if isinstance(object, csc_matrix):
        #         object_id = self.client.put(pickle.dumps(object,
        #                                                  protocol=pickle.HIGHEST_PROTOCOL))
        #     else:
        #         object_id = self.client.put(object)
        #     self.updateStoreInterfaced(object_name, object_id)

        # except SerializationCallbackError:
        #     if isinstance(obj, csc_matrix):  # Ignore rest
        #         object_id = self.client.put(pickle.dumps(obj,
        #                                                  protocol=pickle.HIGHEST_PROTOCOL))

        # except PlasmaObjectExists:
        #     logger.error('Object already exists. Meant to call replace?')
        #     # raise PlasmaObjectExists

        # except ArrowIOError as e:
        #     logger.error('Could not store object '
        #                  + object_name
        #                  + ': {} {}'.format(type(e).__name__, e))
        #     logger.info('Refreshing connection and continuing')
        #     self.reset()

        # except Exception as e:
        #     logger.error('Could not store object '
        #                   + object_name
        #                   + ': {} {}'.format(type(e).__name__, e))

        # if self.use_hdd:
        #     self.lmdb_store.put(obj,
        #                         object_name,
        #                         obj_id=object_id,
        #                         flush_this_immediately=flush_this_immediately)

        # return object_id

    # Before get or getID -
    # check if object is present and sealed (client.contains(obj_id))
    def get(self, object_name):
        """Get a single object from the store
        Checks to see if it knows the object first
        Otherwise throw CannotGetObject to request dict update
        TODO: update for lists of objects
        TODO: replace with getID
        """
        # print('trying to get ', object_name)
        # if self.stored.get(object_name) is None:
        #     logger.error('Never recorded storing this object: '+object_name)
        #     # Don't know anything about this object, treat as problematic
        #     raise CannotGetObjectError(query = object_name)
        # else:
        return self.getID(object_name)

    def getID(self, obj_id, hdd_only=False):
        """
        Get object by object ID

        :param obj_id:
        :type obj_id: class 'plasma.ObjectID'
        :param hdd_only:

        :raises ObjectNotFoundError

        :return: StoreInterfaced object
        """
        # Check in RAM
        if not hdd_only:
            res = self.client.get(obj_id, 0)  # Timeout = 0 ms
            if res is not plasma.ObjectNotAvailable:
                return res if not isinstance(res, bytes) else pickle.loads(res)

        # Check in disk
        if self.use_hdd:
            res = self.lmdb_store.get(obj_id)
            if res is not None:
                return res

        logger.warning("Object {} cannot be found.".format(obj_id))
        raise ObjectNotFoundError

    def getList(self, ids):
        """Get multiple objects from the store"""
        # self._get()
        return self.client.get(ids)

    def get_all(self):
        """Get a listing of all objects in the store"""
        return self.client.list()

    def reset(self):
        """Reset client connection"""
        self.client = self.connect_store(self.store_loc)
        logger.debug("Reset local connection to store: {0}".format(self.store_loc))

    def release(self):
        self.client.disconnect()

    # Necessary? How to fix for functionality?
    # Subscribe to notifications about sealed objects?
    def subscribe(self):
        """Subscribe to a section? of the ds for singals
        Throws unknown errors
        """
        try:
            self.client.subscribe()
        except Exception as e:
            logger.error("Unknown error: {}".format(e))
            raise Exception

    # client.decode_notifications? Get the notification from the buffer?
    # Or we specifically want the next notification from the notification socket?
    # cleint.get_notification_socket first?
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
        """
        self.stored.update({object_name: object_id})

    def getStoreInterfaced(self):
        """returns its info about what it has stored"""
        return self.stored

    def _put(self, obj, id):
        """Internal put"""
        return self.client.put(obj, id)

    def _get(self, object_name):
        """Get an object from the store using its name
        Assumes we know the id for the object_name
        Raises ObjectNotFound if object_id returns no object from the store
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

    # TODO: Likely remove all this functionality for security.
    # def delete(self, id):
    #     try:
    #         self.client.delete([id])
    #     except Exception as e:
    #         logger.error('Couldnt delete: {}'.format(e))

    # Delete below!
    def saveStoreInterface(self, fileName="data/store_dump"):
        """Save the entire store to disk
        Uses pickle, should extend to mmap, hd5f, ...
        """
        raise NotImplementedError

    def saveConfig(self, config_ids, fileName="data/config_dump"):
        """Save current Config object containing parameters
        to run the experiment.
        Config is pickleable
        TODO: move this to Nexus' domain?
        """
        config = self.client.get(config_ids)
        # for object ID in list of items in config, get from store
        # and put into dict (?)
        with open(fileName, "wb") as output:
            pickle.dump(config, output, -1)

    def saveSubstore(self, keys, fileName="data/substore_dump"):
        """Save portion of store based on keys
        to disk
        """
        raise NotImplementedError

    def saveObj(obj, name):
        with open(
            "/media/hawkwings/Ext Hard Drive/dump/dump" + str(name) + ".pkl", "wb"
        ) as output:
            pickle.dump(obj, output)


class LMDBStoreInterface(StoreInterface):
    def __init__(
        self,
        path="../outputs/",
        name=None,
        load=False,
        max_size=1e12,
        flush_immediately=False,
        commit_freq=1,
    ):
        """
        Constructor for LMDB store

        :param path: Path to folder containing LMDB folder.
        :param name: Name of LMDB. Required if not {load].
        :param max_size:
            Maximum size database may grow to; used to size the memory mapping.
            If the database grows larger than map_size, a MapFullError will be raised.
            On 64-bit there is no penalty for making this huge. Must be <2GB on 32-bit.
        :param load: For Replayer use. Informs the class that we're loading
                    from a previous LMDB, not create a new one.
        :param flush_immediately: Save objects to disk immediately
        :param commit_freq: If not flush_immediately,
                            flush data to disk every {commit_freq} seconds.
        """

        # Check if LMDB folder exists.
        # LMDB path?
        path = Path(path)
        if load:
            if name is not None:
                path = path / name
            if not (path / "data.mdb").exists():
                raise FileNotFoundError("Invalid LMDB directory.")
        else:
            assert name is not None
            if not path.exists():
                path.mkdir(parents=True)
            path = path / name

        self.flush_immediately = flush_immediately
        self.lmdb_env = lmdb.open(
            path.as_posix(), map_size=max_size, sync=flush_immediately
        )
        self.lmdb_commit_freq = commit_freq

        self.put_queue = Queue()
        self.put_queue_container = make_dataclass(
            "LMDBPutContainer", [("name", str), ("obj", bytes)]
        )
        # Initialize only after interpreter has forked at the start of each actor.
        self.commit_thread: Thread = None
        signal.signal(signal.SIGINT, self.flush)

    def get(
        self,
        key: Union[plasma.ObjectID, bytes, List[plasma.ObjectID], List[bytes]],
        include_metadata=False,
    ):
        """
        Get object using key (could be any byte string or plasma.ObjectID)

        :param key:
        :param include_metadata: returns whole LMDBData if true else LMDBData.obj
                                (just the stored object).
        :rtype: object or LMDBData
        """
        while True:
            try:
                if isinstance(key, str) or isinstance(key, ObjectID):
                    return self._get_one(
                        LMDBStoreInterface._convert_obj_id_to_bytes(key),
                        include_metadata,
                    )
                return self._get_batch(
                    list(map(LMDBStoreInterface._convert_obj_id_to_bytes, key)),
                    include_metadata,
                )
            except (
                lmdb.BadRslotError
            ):  # Happens when multiple transactions access LMDB at the same time.
                pass

    def _get_one(self, key, include_metadata):
        with self.lmdb_env.begin() as txn:
            r = txn.get(key)

        if r is None:
            return None
        return pickle.loads(r) if include_metadata else pickle.loads(r).obj

    def _get_batch(self, keys, include_metadata):
        with self.lmdb_env.begin() as txn:
            objs = [txn.get(key) for key in keys]

        if include_metadata:
            return [pickle.loads(obj) for obj in objs if obj is not None]
        else:
            return [pickle.loads(obj).obj for obj in objs if obj is not None]

    def get_keys(self):
        """Get all keys in LMDB"""
        with self.lmdb_env.begin() as txn:
            with txn.cursor() as cur:
                cur.first()
                return [key for key in cur.iternext(values=False)]

    def put(self, obj, obj_name, obj_id=None, flush_this_immediately=False):
        """
        Put object ID / object pair into LMDB.

        :param obj: Object to be saved
        :param obj_name:
        :type obj_name: str
        :param obj_id: Object_id from Plasma client
        :type obj_id: class 'plasma.ObjectID'
        :param flush_this_immediately: Override self.flush_immediately.
                                        For storage of critical objects.

        :return: None
        """
        # TODO: Duplication check
        if self.commit_thread is None:
            self.commit_thread = Thread(target=self.commit_daemon, daemon=True)
            self.commit_thread.start()

        if obj_name.startswith("q_") or obj_name.startswith("config"):  # Queue
            name = obj_name.encode()
            is_queue = True
        else:
            name = obj_id.binary() if obj_id is not None else obj_name.encode()
            is_queue = False

        self.put_queue.put(
            self.put_queue_container(
                name=name,
                obj=pickle.dumps(
                    LMDBData(obj, time=time.time(), name=obj_name, is_queue=is_queue)
                ),
            )
        )

        # Write
        if self.flush_immediately or flush_this_immediately:
            self.commit()
            self.lmdb_env.sync()

    def flush(self, sig=None, frame=None):
        """Must run before exiting. Flushes buffer to disk."""
        self.commit()
        self.lmdb_env.sync()
        self.lmdb_env.close()
        exit(0)

    def commit_daemon(self):
        time.sleep(2 * random())  # Reduce multiple commits at the same time.
        while True:
            time.sleep(self.lmdb_commit_freq)
            self.commit()

    def commit(self):
        """Commit objects in {self.put_cache} into LMDB."""
        if not self.put_queue.empty():
            print(self.put_queue.qsize())
            with self.lmdb_env.begin(write=True) as txn:
                while not self.put_queue.empty():
                    container = self.put_queue.get_nowait()
                    txn.put(container.name, container.obj, overwrite=True)

    def delete(self, obj_id):
        """
        Delete object from LMDB.

        :param obj_id:
        :type obj_id: class 'plasma.ObjectID'
        :raises: ObjectNotFoundError
        :return:
        """

        with self.lmdb_env.begin(write=True) as txn:
            out = txn.pop(LMDBStoreInterface._convert_obj_id_to_bytes(obj_id))
        if out is None:
            raise ObjectNotFoundError

    @staticmethod
    def _convert_obj_id_to_bytes(obj_id):
        try:
            return obj_id.binary()
        except AttributeError:
            return obj_id

    def replace(self):
        pass  # TODO

    def subscribe(self):
        pass  # TODO


# Aliasing
StoreInterface = PlasmaStoreInterface


@dataclass
class LMDBData:
    """
    Dataclass to store objects and their metadata into LMDB.
    """

    obj: object
    time: float
    name: str = None
    is_queue: bool = False

    @property
    def queue(self):
        """
        :return: Queue name if object is a queue else None
        """
        # Expected: 'q__Acquirer.q_out__124' -> {'q_out'}
        if self.is_queue:
            try:
                return self.name.split("__")[1].split(".")[1]
            except IndexError:
                return "q_comm"
        logger.error("Attempt to get queue name from objects not from queue.")
        return None


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
