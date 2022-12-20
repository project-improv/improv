import pytest
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

# FIXME: some commented out tests use Limbo --> need to be renamed Store if used

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


def test_connect(setup_store):
    limbo = Store()
    assert isinstance(limbo.client, plasma.PlasmaClient)

def test_connect_incorrect_path(setup_store):
    # TODO: shorter name???
    # TODO: passes, but refactor --- see comments
    store_loc = 'asdf'
    limbo = Store(store_loc)
    # Handle exception thrown - assert name == 'CannotConnectToStoreError' and message == 'Cannot connect to store at {}'.format(str(store_loc))
    # with pytest.raises(Exception, match='CannotConnectToStoreError') as cm:
    #     limbo.connect_store(store_loc)
    #     # Check that the exception thrown is a CannotConnectToStoreError
    #     raise Exception('Cannot connect to store: {0}'.format(e))
    with pytest.raises(CannotConnectToStoreError) as e:
        limbo.connect_store(store_loc)
        # Check that the exception thrown is a CannotConnectToStoreError
    assert e.value.message == 'Cannot connect to store at {}'.format(str(store_loc))

def test_connect_none_path(setup_store):
    # BUT default should be store_loc = '/tmp/store' if not entered?
    store_loc = None
    limbo = Store(store_loc)
    # Handle exception thrown - assert name == 'CannotConnectToStoreError' and message == 'Cannot connect to store at {}'.format(str(store_loc))
    # with pytest.raises(Exception) as cm:
    #     limbo.connnect_store(store_loc)
    # Check that the exception thrown is a CannotConnectToStoreError
    # assert cm.exception.name == 'CannotConnectToStoreError'
    # with pytest.raises(Exception, match='CannotConnectToStoreError') as cm:
    #     limbo.connect_store(store_loc)
    # Check that the exception thrown is a CannotConnectToStoreError
    #     raise Exception('Cannot connect to store: {0}'.format(e))
    with pytest.raises(CannotConnectToStoreError) as e:
        limbo.connect_store(store_loc)
        # Check that the exception thrown is a CannotConnectToStoreError
    assert e.value.message == 'Cannot connect to store at {}'.format(str(store_loc))

# class LimboGet(self):

    # TODO: @pytest.parameterize...limbo.get and limbo.getID for diff datatypes, pickleable and not, etc.
    # Check raises...CannotGetObjectError (object never stored)
def test_init_empty(setup_store):
    limbo = Store()
    assert limbo.get_all() == {}

# class LimboGetID(self):
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
    limbo = Store(store_loc)
    x = limbo.put(mat, 'matrix' )
    assert isinstance(limbo.getID(x), csc_matrix)

# FAILED - ObjectNotFoundError NOT RAISED?
# def test_not_put(setup_store):
#     store_loc = '/tmp/store'
#     limbo = Limbo(store_loc)
#     with pytest.raises(ObjectNotFoundError) as e:
#         obj_id = limbo.getID(limbo.random_ObjectID(1))
#         # Check that the exception thrown is a ObjectNotFoundError
#     assert e.value.message == 'Cannnot find object with ID/name "{}"'.format(obj_id)

# FAILED - AssertionError...looks at LMDBStore in story.py
# assert name is not None?
# def test_use_hdd(setup_store):
#     store_loc = '/tmp/store'
#     limbo = Limbo(store_loc, use_lmdb=True)
#     lmdb_store = limbo.lmdb_store
#     lmdb_store.put(1, 'one')
#     assert lmdb_store.getID('one', hdd_only=True) == 1

# class LimboGetListandAll(StoreDependentTestCase):

@pytest.mark.skip()
def test_get_list_and_all(setup_store):
    limbo = Store()
    id = limbo.put(1, 'one')
    id2 = limbo.put(2, 'two')
    id3 = limbo.put(3, 'three')
    assert [1, 2] == limbo.getList(['one', 'two'])
    assert [1, 2, 3] == limbo.get_all()

# class Limbo_ReleaseReset(StoreDependentTestCase):

# FAILED - DID NOT RAISE <class 'OSError'>???
# def test_release(setup_store):
#     store_loc = '/tmp/store'
#     limbo = Limbo(store_loc)    
#     with pytest.raises(ArrowIOError) as e:
#         limbo.release()
#         limbo.put(1, 'one')
#         # Check that the exception thrown is an ArrowIOError
#     assert e.value.message == 'Could not store object ' + object_name + ': {} {}'.format(type(e).__name__, e)
#     # TODO: assert info == 'Refreshing connection and continuing'

def test_reset(setup_store):
    limbo = Store()
    limbo.reset()
    id = limbo.put(1, 'one')
    assert limbo.get(id) == 1

# class Limbo_Put(StoreDependentTestCase):

def test_put_one(setup_store):
    limbo = Store()
    id = limbo.put(1, 'one')
    assert 1 == limbo.get(id)

@pytest.mark.skip(reason = 'Error not being raised')
def test_put_twice(setup_store):
    limbo = Store()
    with pytest.raises(PlasmaObjectExists) as e:
        id = limbo.put(2, 'two')
        id2 = limbo.put(2, 'two')
        # Check that the exception thrown is an PlasmaObjectExists
    assert e.value.message == 'Object already exists. Meant to call replace?'

# class Limbo_PutGet(StoreDependentTestCase):

def test_getOne(setup_store):
    limbo = Store()
    id = limbo.put(1, 'one')
    id2 = limbo.put(2, 'two')
    assert 1 == limbo.get(id)

# def test_get_nonexistent(setup_store):
#     limbo = Limbo()
#     # Handle exception thrown
#     # Check that the exception thrown is a CannotGetObjectError
#     with pytest.raises(CannotGetObjectError) as e:
#         # Check that the exception thrown is an PlasmaObjectExists
#         limbo.get('three')
#         assert e.value.message == 'Cannot get object {}'.format(self.query)

### TODO:
"""class Limbo_Notify(StoreDependentTestCase):

    def test_notify(self):
        # TODO: not unit testable?

### This is NOT USED anymore???
class Limbo_UpdateStored(StoreDependentTestCase):

    # Accessing self.limbo.stored directly to test getStored separately
    def test_updateGet(self):
        self.limbo.put(1, 'one')
        self.limbo.updateStored('one', 3)
        assert 3 == self.limbo.stored['one']

class Limbo_GetStored(StoreDependentTestCase):

    def test_getStoredEmpty(self):
        assert self.limbo.getStored() == False

    def test_putGetStored(self):
        self.limbo.put(1, 'one')
        assert 1 == self.limbo.getID(self.limbo.getStored()['one'])

class Limbo_internalPutGet(StoreDependentTestCase):

    def test_put(self):
        id = self.limbo.random_ObjectID(1)
        self.limbo._put(1, id[0])
        assert 1 == self.limbo.client.get(id[0])

    def test_get(self):
        id= self.limbo.put(1, 'one')
        self.limbo.updateStored('one', id)
        assert self.limbo._get('one') == 1

    def test__getNonexistent(self):

        # Handle exception thrown
        with pytest.raises(Exception) as cm:
        # Check that the exception thrown is a ObjectNotFoundError
            self.limbo._get('three')
            assert cm.exception.name == 'ObjectNotFoundError'
            assert cm.exception.message == 'Cannnot find object with ID/name "three"'

class Limbo_saveTweak(StoreDependentTestCase):

    def test_tweak(self):
        fileName= 'data/tweak_dump'
        id= self.limbo.put(1, 'one')
        id2= self.limbo.put(2, 'two')
        tweak_ids=[id, id2]
        self.limbo.saveTweak(tweak_ids)
        with open(fileName, 'rb') as output:
            assert pickle.load(output) == [1, 2]

# Test out CSC matrix format after updating to arrow 0.14.0
class Limbo_sparseMatrix(StoreDependentTestCase):

    def test_csc(self):
        csc = csc_matrix((3, 4), dtype=np.int8)
        self.limbo.put(csc, "csc")
        assert np.allclose(self.limbo.get("csc").toarray(), csc.toarray()) == True"""