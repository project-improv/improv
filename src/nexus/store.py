import datetime
import pickle
import signal
import time

from dataclasses import dataclass, make_dataclass
from queue import Empty, Queue
from pathlib import Path
from random import random
from threading import Thread
from typing import List, Union

import lmdb
import numpy as np
import pyarrow.plasma as plasma
from pyarrow import PlasmaObjectExists, SerializationCallbackError
from pyarrow.lib import ArrowIOError
from pyarrow.plasma import ObjectNotAvailable, ObjectID
from scipy.sparse import csc_matrix

from nexus.actor import Spike

import logging; logger=logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

#TODO: Use Apache Arrow for better memory usage with the Plasma store

class StoreInterface():
    '''General interface for a store
    '''
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


class Limbo(StoreInterface):
    ''' Basic interface for our specific data store
        implemented with apache arrow plasma
        Objects are stored with object_ids
        References to objects are contained in a dict where key is
          shortname, value is object_id
    '''

    def __init__(self, name='default', store_loc='/tmp/store',
                 use_hdd=False, lmdb_name=None, hdd_maxstore=1e12, hdd_path='output/', flush_immediately=False,
                 commit_freq=1):

        """
        Constructor for the Limbo

        :param name:
        :param store_loc: Apache Arrow Plasma client location

        :param use_hdd: bool Also write data to disk using the LMDB

        :param hdd_maxstore:
            Maximum size database may grow to; used to size the memory mapping.
            If the database grows larger than map_size, a MapFullError will be raised.
            On 64-bit there is no penalty for making this huge. Must be <2GB on 32-bit.

        :param hdd_path: Path to LMDB folder.
        :param flush_immediately: Save objects to disk immediately
        :param commit_freq: If not flush_immediately, flush data to disk every {commit_freq} seconds.
        """

        self.name = name
        self.store_loc = store_loc
        self.client = self.connectStore(store_loc)
        self.stored = {}

        # Offline db
        self.use_hdd = use_hdd
        self.flush_immediately = flush_immediately

        if use_hdd:
            self.lmdb_store = LMDBStore(name=lmdb_name, max_size=hdd_maxstore, path=hdd_path,
                                        flush_immediately=flush_immediately,
                                        commit_freq=commit_freq)

    def connectStore(self, store_loc):
        ''' Connect to the store at store_loc
            Raises exception if can't connect
            Returns the plamaclient if successful
            Updates the client internal
        '''
        try:
            #self.client = plasma.connect(store_loc)
            self.client = plasma.connect(store_loc, '', 0)
            logger.info('Successfully connected to store')
        except Exception as e:
            logger.exception('Cannot connect to store: {0}'.format(e))
            raise Exception
        return self.client

    def put(self, obj, object_name, flush_this_immediately=False):
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
            object_id = self.client.put(obj)
            self.updateStored(object_name, object_id)

        except SerializationCallbackError:
            if isinstance(obj, csc_matrix):  # Ignore rest
                object_id = self.client.put(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))

        except PlasmaObjectExists:
            logger.error('Object already exists. Meant to call replace?')
            # raise PlasmaObjectExists

        except ArrowIOError as e:
            logger.error('Could not store object ' + object_name + ': {} {}'.format(type(e).__name__, e))
            logger.info('Refreshing connection and continuing')
            self.reset()

        except Exception as e:
            logger.error('Could not store object ' + object_name + ': {} {}'.format(type(e).__name__, e))

        if self.use_hdd:
            self.lmdb_store.put(obj, object_name, obj_id=object_id, flush_this_immediately=flush_this_immediately)

        return object_id

    def get(self, object_name):
        ''' Get a single object from the store
            Checks to see if it knows the object first
            Otherwise throw CannotGetObject to request dict update
            TODO: update for lists of objects
        '''
        #print('trying to get ', object_name)
        if self.stored.get(object_name) is None:
            logger.error('Never recorded storing this object: '+object_name)
            # Don't know anything about this object, treat as problematic
            raise CannotGetObjectError
        else:
            return self._get(object_name)
    
    def getID(self, obj_id, hdd_only=False):
        """
        Get object by object ID

        :param obj_id:
        :type obj_id: class 'plasma.ObjectID'
        :param hdd_only:

        :raises ObjectNotFoundError

        :return: Stored object
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

        logger.warning('Object {} cannot be found.'.format(obj_id))
        raise ObjectNotFoundError

    def getList(self, ids):
        """
        Get objects using a list of object ID

        :param ids: List of object IDs
        :type ids: List[plasma.ObjectID]
        :return: List of requested objects
        :rtype: List[object]
        """
        return [self.getID(i) for i in ids]

    def get_all(self):
        ''' Get a listing of all objects in the store
        '''
        return self.client.list()

    def reset(self):
        ''' Reset client connection
        '''
        self.client = self.connectStore(self.store_loc)
        logger.debug('Reset local connection to store')

    def release(self):
        self.client.disconnect()

    def subscribe(self):
        ''' Subscribe to a section? of the ds for singals
            Throws unknown errors
        '''
        try: 
            self.client.subscribe()
        except Exception as e:
            logger.error('Unknown error: {}'.format(e))
            raise Exception

    def notify(self):
        try:
            notification_info = self.client.get_next_notification()
            #recv_objid, recv_dsize, recv_msize = notification_info
        except ArrowIOError as e:
            notification_info = None
        except Exception as e:
            logger.exception('Notification error: {}'.format(e))
            raise Exception

        return notification_info
    
    def random_ObjectID(self, number=1):
        ''' Convenience function if ID needs to be specified first.
        '''
        ids = []
        for i in range(number):
            ids.append(plasma.ObjectID(np.random.bytes(20)))
        return ids

    def updateStored(self, object_name, object_id):
        ''' Update local dict with info we need locally
            Report to Nexus that we updated the store
                (did a put or delete/replace)
        '''
        self.stored.update({object_name:object_id})

    def getStored(self):
        ''' returns its info about what it has stored
        '''
        return self.stored
    
    def _put(self, obj, id):
        ''' Internal put
        '''
        return self.client.put(obj, id)    

    def _get(self, object_name):
        ''' Get an object from the store using its name
            Assumes we know the id for the object_name
            Raises ObjectNotFound if object_id returns no object from the store
        '''
        res = self.client.get(self.stored.get(object_name), 0)
        # Can also use contains() to check
        if isinstance(res, ObjectNotAvailable):
            logger.warning('Object {} cannot be found.'.format(object_name))
            raise ObjectNotFoundError #TODO: Don't raise?
        else:
            return res

    #TODO: Likely remove all this functionality for security.
    # def deleteName(self, object_name):
    #     ''' Deletes an object from the store based on name
    #         assumes we have id from name
    #         This prevents us from deleting other portions of 
    #         the store that we don't have access to
    #     '''

    #     if self.stored.get(object_name) is None:
    #         logger.error('Never recorded storing this object: '+object_name)
    #         # Don't know anything about this object, treat as problematic
    #         raise CannotGetObjectError
    #     else:
    #         retcode = self._delete(object_name)
    #         self.stored.pop(object_name)
            
    # def delete(self, id):
    #     try:
    #         self.client.delete([id])
    #     except Exception as e:
    #         logger.error('Couldnt delete: {}'.format(e))
    
    # def _delete(self, object_name):
    #     ''' Deletes object from store
    #     '''
    #     tmp_id = self.stored.get(object_name)

    #     new_client = plasma.connect('/tmp/store', '', 0)
    #     new_client.delete([tmp_id])
    #     new_client.disconnect()

    def saveStore(self, fileName='/home/store_dump'):
        ''' Save the entire store to disk
            Uses pickle, should extend to mmap, hd5f, ...
        '''
        raise NotImplementedError

    def saveTweak(self, tweak_ids, fileName='/home/tweak_dump'):
        ''' Save current Tweak object containing parameters
            to run the experiment.
            Tweak is pickleable
            TODO: move this to Nexus' domain?
        '''
        tweak = self.client.get(tweak_ids)
        #for object ID in list of items in tweak, get from store
        #and put into dict (?)
        with open(fileName, 'wb') as output:
            pickle.dump(tweak, output, -1)

    def saveSubstore(self, keys, fileName='/home/substore_dump'):
        ''' Save portion of store based on keys
            to disk
        '''
        raise NotImplementedError


class LMDBStore(StoreInterface):
    def __init__(self, path='../outputs/', name=None, load=False, max_size=1e12,
                 flush_immediately=False, commit_freq=1):
        """
        Constructor for LMDB store

        :param path: Path to LMDB folder.
        :param name: Name of LMDB. Default to lmdb_[current time in human format].
        :param max_size:
            Maximum size database may grow to; used to size the memory mapping.
            If the database grows larger than map_size, a MapFullError will be raised.
            On 64-bit there is no penalty for making this huge. Must be <2GB on 32-bit.
        :param load: For Replayer use. Informs the class that we're loading from a previous LMDB, not create a new one.
        :param flush_immediately: Save objects to disk immediately
        :param commit_freq: If not flush_immediately, flush data to disk every {commit_freq} seconds.
        """

        # Check if LMDB folder exists.
        path = Path(path)
        if load:
            if name is not None:
                path = path / name
            if not (path / 'data.mdb').exists():
                raise FileNotFoundError('Invalid LMDB directory.')
        else:
            if not path.exists():
                path.mkdir(parents=True)
            if name is None:
                name = f'lmdb_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
            path = path / name

        self.flush_immediately = flush_immediately
        self.lmdb_env = lmdb.open(path.as_posix(), map_size=max_size, sync=flush_immediately)
        self.lmdb_commit_freq = commit_freq

        self.put_queue = Queue()
        self.put_queue_container = make_dataclass('LMDBPutContainer', [('name', str), ('obj', bytes)])

        self.commit_thread = None  # Initialize only after interpreter has forked at the start of each actor.
        signal.signal(signal.SIGINT, self.flush)

    def get(self, key: Union[plasma.ObjectID, bytes, List[plasma.ObjectID], List[bytes]], include_metadata=False):
        """
        Get object using key (could be any byte string or plasma.ObjectID)

        :param key:
        :param include_metadata: returns whole LMDBData if true else LMDBData.obj (just the stored object).
        :rtype: object or LMDBData
        """
        while True:
            try:
                if isinstance(key, str) or isinstance(key, ObjectID):
                    return self._get_one(LMDBStore._convert_obj_id_to_bytes(key), include_metadata)
                return self._get_batch(list(map(LMDBStore._convert_obj_id_to_bytes, key)), include_metadata)
            except lmdb.BadRslotError:  # Happens when multiple transactions access LMDB at the same time.
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
        """ Get all keys in LMDB """
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
        :param flush_this_immediately: Override self.flush_immediately. For storage of critical objects.

        :return: None
        """
        # TODO: Duplication check
        if self.commit_thread is None:
            self.commit_thread = Thread(target=self.commit_daemon, daemon=True)
            self.commit_thread.start()

        if obj_name.startswith('q_') or obj_name.startswith('tweak'):  # Queue
            name = obj_name.encode()
            is_queue = True
        else:
            name = obj_id.binary() if obj_id is not None else obj_name.encode()
            is_queue = False

        self.put_queue.put(self.put_queue_container(
               name=name, obj=pickle.dumps(LMDBData(obj, time=time.time(), name=obj_name, is_queue=is_queue))))

        # Write
        if self.flush_immediately or flush_this_immediately:
            self.commit()
            self.lmdb_env.sync()

    def flush(self, sig=None, frame=None):
        """ Must run before exiting. Flushes buffer to disk. """
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
        """ Commit objects in {self.put_cache} into LMDB. """
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
            out = txn.pop(LMDBStore._convert_obj_id_to_bytes(obj_id))
        if out is None:
            raise ObjectNotFoundError

    def replace(self): pass

    def subscribe(self): pass

    @staticmethod
    def _convert_obj_id_to_bytes(obj_id):
        try:
            return obj_id.binary()
        except AttributeError:
            return obj_id


def saveObj(obj, name):
    with open('/media/hawkwings/Ext Hard Drive/dump/dump'+str(name)+'.pkl', 'wb') as output:
        pickle.dump(obj, output)


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
                return self.name.split('__')[1].split('.')[1]
            except IndexError:
                return 'q_comm'
        logger.error('Attempt to get queue name from objects not from queue.')
        return None

class ObjectNotFoundError(Exception):
    pass

class CannotGetObjectError(Exception):
    pass


class Watcher():
    ''' Monitors the store as separate process
        TODO: Facilitate Watcher being used in multiple processes (shared list)
    '''
    def __init__(self, name, client):
        self.name = name
        self.client = client
        self.flag = False
        self.saved_ids = []

        self.client.subscribe()
        self.n = 0

    def setLinks(self, links):
        self.q_sig = links

    def run(self):
        while True:
            if self.flag:
                try:
                    self.checkStore2()
                except Exception as e:
                    logger.error('Watcher exception during run: {}'.format(e))
                    #break 
            try: 
                signal = self.q_sig.get(timeout=0.005)
                if signal == Spike.run(): 
                    self.flag = True
                    logger.warning('Received run signal, begin running')
                elif signal == Spike.quit():
                    logger.warning('Received quit signal, aborting')
                    break
                elif signal == Spike.pause():
                    logger.warning('Received pause signal, pending...')
                    self.flag = False
                elif signal == Spike.resume(): #currently treat as same as run
                    logger.warning('Received resume signal, resuming')
                    self.flag = True
            except Empty as e:
                pass #no signal from Nexus

    # def checkStore(self):
    #     notification_info = self.client.notify()
    #     recv_objid, recv_dsize, recv_msize = notification_info
    #     obj = self.client.getID(recv_objid)
    #     try:
    #         self.saveObj(obj)
    #         self.n += 1
    #     except Exception as e:
    #         logger.error('Watcher error: {}'.format(e))

    def saveObj(self, obj, name):
        with open('/media/hawkwings/Ext Hard Drive/dump/dump'+name+'.pkl', 'wb') as output:
            pickle.dump(obj, output)

    def checkStore2(self):
        objs = list(self.client.get_all().keys())
        ids_to_save = list(set(objs) - set(self.saved_ids))

        # with Pool() as pool:
        #     saved_ids = pool.map(saveObjbyID, ids_to_save)
        # print('Saved :', len(saved_ids))
        # self.saved_ids.extend(saved_ids)

        for id in ids_to_save:
            self.saveObj(self.client.getID(id), str(id))
            self.saved_ids.append(id)

# def saveObjbyID(id):
#     client = plasma.connect('/tmp/store')
#     obj = client.get(id)
#     with open('/media/hawkwings/Ext\ Hard\ Drive/dump/dump'+str(id)+'.pkl', 'wb') as output:
#         pickle.dump(obj, output)
#     return id
            
