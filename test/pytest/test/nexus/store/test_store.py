import pytest
from improv.store import Limbo
from multiprocessing import Process
from pyarrow import PlasmaObjectExists
from scipy.sparse import csc_matrix
import numpy as np
import pyarrow.plasma as plasma
from pyarrow.lib import ArrowIOError
from improv.store import ObjectNotFoundError
from improv.store import CannotGetObjectError
from improv.store import CannotConnectToStoreError
import pickle

# Separate each class as individual file - individual tests???
# TODO: remove set_up and tear_down - test if behavior is the same w/fixtures

@pytest.fixture(scope="function")
# TODO: change name...
# def run_before_after?
def limbo():
    # self.limbo = Limbo()
    return Limbo()

class LimboConnect(self):

    def test_connect(self):
        store_loc = '/tmp/store'
        self.limbo.connect_store(store_loc)
        assert isinstance(self.limbo.client, plasma.PlasmaClient)

    def test_connect_incorrect_path(self):
        # TODO: shorter name???
        store_loc = 'asdf'
        # Handle exception thrown
        with pytest.raises(Exception) as cm:
            self.limbo.connnect_store(store_loc)
        # Check that the exception thrown is a CannotConnectToStoreError
        assert cm.exception.name == 'CannotConnectToStoreError'

    def test_connect_none_path(self):
        # BUT default should be store_loc = '/tmp/store' if not entered?
        store_loc = None
        # Handle exception thrown
        with pytest.raises(Exception) as cm:
            self.limbo.connnect_store(store_loc)
        # Check that the exception thrown is a CannotConnectToStoreError
        assert cm.exception.name == 'CannotConnectToStoreError'

class LimboGet(self):
    # TODO: @pytest.parameterize...limbo.get and limbo.getID for diff datatypes, pickleable and not, etc.
    # Check raises...CannotGetObjectError (object never stored)
    def test_init_empty(self):
        assert self.limbo.get_all() == False

class LimboGetID(self):
    # TODO:
    # Check both hdd_only=False/True
    # Check isInstance type, isInstance bytes, else
    # Check in disk - pytest.raises(ObjectNotFoundError)
    # Decide to test_getList and test_get_all

    def test_is_picklable(self):
        # Test if obj to put is picklable - if not raise error, handle/suggest how to fix

    def test_is_csc_matrix_and_put(self):
        mat = csc_matrix((3, 4), dtype=np.int8)
        x = self.limbo.put(mat, 'matrix' )
        assert isinstance(self.limbo.getID(x), csc_matrix)

    def test_not_put(self):
        obj = self.limbo.getID(self.limbo.random_ObjectID(1))
        pytest.raises(ObjectNotFoundError)

    def test_use_hdd(self):
        self.lmdb_store.put(1, 'one')
        
        assert self.lmdb_store.getID('one', hdd_only=True) == 1

class LimboGetListandAll(StoreDependentTestCase):

    def getListandAll(self):
        id = self.limbo.put(1, 'one')
        id2 = self.limbo.put(2, 'two')
        id3 = self.limbo.put(3, 'three')
        self.assertEqual([1, 2], self.limbo.getList(['one', 'two']))
        self.assertEqual([1, 2, 3], self.limbo.get_all())

class Limbo_ReleaseReset(StoreDependentTestCase):

    def test_release(self):
        self.limbo.release()
        self.limbo.put(1, 'one')
        self.assertRaises(ArrowIOError)

    def test_reset(self):
        self.limbo.reset()
        self.limbo.put(1, 'one')
        self.assertEqual(self.limbo.get('one'), 1)

class Limbo_Put(StoreDependentTestCase):

    def test_putOne(self):
        id = self.limbo.put(1, 'one')
        self.assertEqual(1, self.limbo.get('one'))

    def test_put_twice(self):
        id = self.limbo.put(2, 'two')
        id2 = self.limbo.put(2, 'two')
        self.assertRaises(PlasmaObjectExists)

class Limbo_PutGet(StoreDependentTestCase):

    def test_getOne(self):
        id = self.limbo.put(1, 'one')
        id2 = self.limbo.put(2, 'two')
        self.assertEqual(1, self.limbo.get('one'))
        self.assertEqual(id, self.limbo.stored['one'])

    def test_get_nonexistent(self):

        # Handle exception thrown
        with self.assertRaises(Exception) as cm:
            self.limbo.get('three')

        # Check that the exception thrown is a CannotGetObjectError
        self.assertEqual(cm.exception.name, 'CannotGetObjectError')

"""class Limbo_Notify(StoreDependentTestCase):

    # Add test body here
    def test_notify(self):
        # TODO: not unit testable?


class Limbo_UpdateStored(StoreDependentTestCase):

    # Accessing self.limbo.stored directly to test getStored separately
    def test_updateGet(self):
        self.limbo.put(1, 'one')
        self.limbo.updateStored('one', 3)
        self.assertEqual(3,self.limbo.stored['one'])

class Limbo_GetStored(StoreDependentTestCase):

    def test_getStoredEmpty(self):
        self.assertFalse(self.limbo.getStored())

    def test_putGetStored(self):
        self.limbo.put(1, 'one')
        self.assertEqual(1, self.limbo.getID(self.limbo.getStored()['one'])) # returns ID

class Limbo_internalPutGet(StoreDependentTestCase):

    def test_put(self):
        id = self.limbo.random_ObjectID(1)
        self.limbo._put(1, id[0])
        self.assertEqual(1, self.limbo.client.get(id[0]))

    def test_get(self):
        id= self.limbo.put(1, 'one')
        self.limbo.updateStored('one', id)
        self.assertEqual(self.limbo._get('one'), 1)

    def test__getNonexistent(self):

        # Handle exception thrown
        with self.assertRaises(Exception) as cm:
        # Check that the exception thrown is a ObjectNotFoundError
            self.limbo._get('three')
            self.assertEqual(cm.exception.name, 'ObjectNotFoundError')
            self.assertEqual(cm.exception.message, 'Cannnot find object with ID/name "three"')

class Limbo_saveTweak(StoreDependentTestCase):

    def test_tweak(self):
        fileName= 'data/tweak_dump'
        id= self.limbo.put(1, 'one')
        id2= self.limbo.put(2, 'two')
        tweak_ids=[id, id2]
        self.limbo.saveTweak(tweak_ids)
        with open(fileName, 'rb') as output:
            self.assertEqual(pickle.load(output), [1, 2])

# Test out CSC matrix format after updating to arrow 0.14.0
class Limbo_sparseMatrix(StoreDependentTestCase):

    def test_csc(self):
        csc = csc_matrix((3, 4), dtype=np.int8)
        self.limbo.put(csc, "csc")
        self.assertTrue(np.allclose(self.limbo.get("csc").toarray(), csc.toarray()))