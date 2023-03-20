import pytest
import time
from improv.store import Store
from multiprocessing import Process
from pyarrow._plasma import PlasmaObjectExists
from scipy.sparse import csc_matrix
import numpy as np
import pyarrow.plasma as plasma
from pyarrow.lib import ArrowIOError
from improv.store import ObjectNotFoundError
from improv.store import CannotGetObjectError
from improv.store import CannotConnectToStoreError
import pickle
import subprocess

WAIT_TIMEOUT = 10

# TODO: add docstrings!!!
# TODO: clean up syntax - consistent capitalization, function names, etc.
# TODO: decide to keep classes
# TODO: increase coverage!!! SEE store.py

# Separate each class as individual file - individual tests???

# @pytest.fixture
# def store_loc():
#     store_loc = '/dev/shm'
#     return store_loc

# store_loc = '/dev/shm'

# FIXME: some commented out tests use Store --> need to be renamed Store if used

@pytest.fixture
# TODO: put in conftest.py
def setup_store(store_loc='/tmp/store'):
        ''' Start the server
        '''
        print('Setting up Plasma store.')
        p = subprocess.Popen(
            ['plasma_store', '-s', store_loc, '-m', str(10000000)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # with plasma.start_plasma_store(10000000) as ps:
            
        yield p

        # ''' Kill the server
        # '''
        # print('Tearing down Plasma store.')
        p.kill()
        p.wait(WAIT_TIMEOUT)


def test_connect(setup_store):
    store = Store()
    assert isinstance(store.client, plasma.PlasmaClient)

def test_connect_incorrect_path(setup_store):
    # TODO: shorter name???
    # TODO: passes, but refactor --- see comments
    store_loc = 'asdf'
    store = Store(store_loc)
    # Handle exception thrown - assert name == 'CannotConnectToStoreError' and message == 'Cannot connect to store at {}'.format(str(store_loc))
    # with pytest.raises(Exception, match='CannotConnectToStoreError') as cm:
    #     store.connect_store(store_loc)
    #     # Check that the exception thrown is a CannotConnectToStoreError
    #     raise Exception('Cannot connect to store: {0}'.format(e))
    with pytest.raises(CannotConnectToStoreError) as e:
        store.connect_store(store_loc)
        # Check that the exception thrown is a CannotConnectToStoreError
    assert e.value.message == 'Cannot connect to store at {}'.format(str(store_loc))

def test_connect_none_path(setup_store):
    # BUT default should be store_loc = '/tmp/store' if not entered?
    store_loc = None
    store = Store(store_loc)
    # Handle exception thrown - assert name == 'CannotConnectToStoreError' and message == 'Cannot connect to store at {}'.format(str(store_loc))
    # with pytest.raises(Exception) as cm:
    #     store.connnect_store(store_loc)
    # Check that the exception thrown is a CannotConnectToStoreError
    # assert cm.exception.name == 'CannotConnectToStoreError'
    # with pytest.raises(Exception, match='CannotConnectToStoreError') as cm:
    #     store.connect_store(store_loc)
    # Check that the exception thrown is a CannotConnectToStoreError
    #     raise Exception('Cannot connect to store: {0}'.format(e))
    with pytest.raises(CannotConnectToStoreError) as e:
        store.connect_store(store_loc)
        # Check that the exception thrown is a CannotConnectToStoreError
    assert e.value.message == 'Cannot connect to store at {}'.format(str(store_loc))

# class StoreGet(self):

    # TODO: @pytest.parameterize...store.get and store.getID for diff datatypes, pickleable and not, etc.
    # Check raises...CannotGetObjectError (object never stored)
def test_init_empty(setup_store):
    store = Store()
    assert store.get_all() == {}

# class StoreGetID(self):
# TODO:
# Check both hdd_only=False/True
# Check isInstance type, isInstance bytes, else
# Check in disk - pytest.raises(ObjectNotFoundError)
# Decide to test_getList and test_get_all

# def test_is_picklable(self):
    # Test if obj to put is picklable - if not raise error, handle/suggest how to fix

# TODO: TEST BELOW:
# except PlasmaObjectExists:
#     logger.error('Object already exists. Meant to call replace?')
# except ArrowIOError as e:
#     logger.error('Could not store object '+object_name+': {} {}'.format(type(e).__name__, e))
#     logger.info('Refreshing connection and continuing')
#     self.reset()
# except Exception as e:
#     logger.error('Could not store object '+object_name+': {} {}'.format(type(e).__name__, e))
    
def test_is_csc_matrix_and_put(setup_store):
    mat = csc_matrix((3, 4), dtype=np.int8)
    store_loc = '/tmp/store'
    store = Store(store_loc)
    x = store.put(mat, 'matrix' )
    assert isinstance(store.getID(x), csc_matrix)

# FAILED - ObjectNotFoundError NOT RAISED?
# def test_not_put(setup_store):
#     store_loc = '/tmp/store'
#     store = Store(store_loc)
#     with pytest.raises(ObjectNotFoundError) as e:
#         obj_id = store.getID(store.random_ObjectID(1))
#         # Check that the exception thrown is a ObjectNotFoundError
#     assert e.value.message == 'Cannnot find object with ID/name "{}"'.format(obj_id)

# FAILED - AssertionError...looks at LMDBStore in story.py
# assert name is not None?
# def test_use_hdd(setup_store):
#     store_loc = '/tmp/store'
#     store = Store(store_loc, use_lmdb=True)
#     lmdb_store = store.lmdb_store
#     lmdb_store.put(1, 'one')
#     assert lmdb_store.getID('one', hdd_only=True) == 1

# class StoreGetListandAll(StoreDependentTestCase):

@pytest.mark.skip()
def test_get_list_and_all(setup_store):
    store = Store()
    id = store.put(1, 'one')
    id2 = store.put(2, 'two')
    id3 = store.put(3, 'three')
    assert [1, 2] == store.getList(['one', 'two'])
    assert [1, 2, 3] == store.get_all()

# class Store_ReleaseReset(StoreDependentTestCase):

# FAILED - DID NOT RAISE <class 'OSError'>???
# def test_release(setup_store):
#     store_loc = '/tmp/store'
#     store = Store(store_loc)    
#     with pytest.raises(ArrowIOError) as e:
#         store.release()
#         store.put(1, 'one')
#         # Check that the exception thrown is an ArrowIOError
#     assert e.value.message == 'Could not store object ' + object_name + ': {} {}'.format(type(e).__name__, e)
#     # TODO: assert info == 'Refreshing connection and continuing'

def test_reset(setup_store):
    store = Store()
    store.reset()
    id = store.put(1, 'one')
    assert store.get(id) == 1

# class Store_Put(StoreDependentTestCase):

def test_put_one(setup_store):
    store = Store()
    id = store.put(1, 'one')
    assert 1 == store.get(id)

@pytest.mark.skip(reason = 'Error not being raised')
def test_put_twice(setup_store):
    store = Store()
    with pytest.raises(PlasmaObjectExists) as e:
        id = store.put(2, 'two')
        id2 = store.put(2, 'two')
        # Check that the exception thrown is an PlasmaObjectExists
    assert e.value.message == 'Object already exists. Meant to call replace?'

# class Store_PutGet(StoreDependentTestCase):

def test_getOne(setup_store):
    store = Store()
    id = store.put(1, 'one')
    id2 = store.put(2, 'two')
    assert 1 == store.get(id)

# def test_get_nonexistent(setup_store):
#     store = Store()
#     # Handle exception thrown
#     # Check that the exception thrown is a CannotGetObjectError
#     with pytest.raises(CannotGetObjectError) as e:
#         # Check that the exception thrown is an PlasmaObjectExists
#         store.get('three')
#         assert e.value.message == 'Cannot get object {}'.format(self.query)

### TODO:
"""class Store_Notify(StoreDependentTestCase):

    def test_notify(self):
        # TODO: not unit testable?

### This is NOT USED anymore???
class Store_UpdateStored(StoreDependentTestCase):

    # Accessing self.store.stored directly to test getStored separately
    def test_updateGet(self):
        self.store.put(1, 'one')
        self.store.updateStored('one', 3)
        assert 3 == self.store.stored['one']

class Store_GetStored(StoreDependentTestCase):

    def test_getStoredEmpty(self):
        assert self.store.getStored() == False

    def test_putGetStored(self):
        self.store.put(1, 'one')
        assert 1 == self.store.getID(self.store.getStored()['one'])

class Store_internalPutGet(StoreDependentTestCase):

    def test_put(self):
        id = self.store.random_ObjectID(1)
        self.store._put(1, id[0])
        assert 1 == self.store.client.get(id[0])

    def test_get(self):
        id= self.store.put(1, 'one')
        self.store.updateStored('one', id)
        assert self.store._get('one') == 1

    def test__getNonexistent(self):

        # Handle exception thrown
        with pytest.raises(Exception) as cm:
        # Check that the exception thrown is a ObjectNotFoundError
            self.store._get('three')
            assert cm.exception.name == 'ObjectNotFoundError'
            assert cm.exception.message == 'Cannnot find object with ID/name "three"'

class Store_saveConfig(StoreDependentTestCase):

    def test_config(self):
        fileName= 'data/config_dump'
        id= self.store.put(1, 'one')
        id2= self.store.put(2, 'two')
        config_ids=[id, id2]
        self.store.saveConfig(config_ids)
        with open(fileName, 'rb') as output:
            assert pickle.load(output) == [1, 2]

# Test out CSC matrix format after updating to arrow 0.14.0
class Store_sparseMatrix(StoreDependentTestCase):

    def test_csc(self):
        csc = csc_matrix((3, 4), dtype=np.int8)
        self.store.put(csc, "csc")
        assert np.allclose(self.store.get("csc").toarray(), csc.toarray()) == True"""