from unittest import TestCase
from test.test_utils import StoreDependentTestCase
from improv.store import Store
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


class Store_Connect(StoreDependentTestCase):
    def setUp(self):
        super(Store_Connect, self).setUp()
        self.store = Store()

    def test_Connect(self):
        store_loc = "/tmp/store"
        self.store.connectStore(store_loc)
        self.assertIsInstance(self.store.client, plasma.PlasmaClient)

    def test_failToConnect(self):
        store_loc = "asdf"

        # Handle exception thrown
        with self.assertRaises(Exception) as cm:
            self.store.connectStore(store_loc)

        # Check that the exception thrown is a CannotConnectToStoreError
        self.assertEqual(cm.exception.name, "CannotConnectToStoreError")

    def tearDown(self):
        super(Store_Connect, self).tearDown()


class Store_Get(StoreDependentTestCase):
    def setUp(self):
        super(Store_Get, self).setUp()
        self.store = Store()

    def test_init_empty(self):
        self.assertFalse(self.store.get_all())

    def tearDown(self):
        super(Store_Get, self).tearDown()


class Store_GetID(StoreDependentTestCase):
    # Check both hdd_only=False/True
    # Check isInstance type, isInstance bytes, else

    def setUp(self):
        super(Store_GetID, self).setUp()
        self.store = Store()

    def test_isMatrix(self):  # also tests put matrix
        mat = csc_matrix((3, 4), dtype=np.int8)
        x = self.store.put(mat, "matrix")  # returns object_id
        self.assertIsInstance(self.store.getID(x), csc_matrix)

    def test_notPut(self):
        obj = self.store.getID(self.store.random_ObjectID(1))
        self.assertRaises(ObjectNotFoundError)

    def UseHDD(self):
        self.lmdb_store.put(1, "one")
        assertEqual(self.lmdb_store.getID("one", hdd_only=True), 1)

    def tearDown(self):
        super(Store_GetID, self).tearDown()


class Store_getListandAll(StoreDependentTestCase):
    def setUp(self):
        super(Store_getListandAll, self).setUp()
        self.store = Store()

    def getListandAll(self):
        id = self.store.put(1, "one")
        id2 = self.store.put(2, "two")
        id3 = self.store.put(3, "three")
        self.assertEqual([1, 2], self.store.getList(["one", "two"]))
        self.assertEqual([1, 2, 3], self.store.get_all())

    def tearDown(self):
        super(Store_getListandAll, self).tearDown()


class Store_ReleaseReset(StoreDependentTestCase):
    def setUp(self):
        super(Store_ReleaseReset, self).setUp()
        self.store = Store()

    def test_release(self):
        self.store.release()
        self.store.put(1, "one")
        self.assertRaises(ArrowIOError)

    def test_reset(self):
        self.store.reset()
        self.store.put(1, "one")
        self.assertEqual(self.store.get("one"), 1)

    def tearDown(self):
        super(Store_ReleaseReset, self).tearDown()


class Store_Put(StoreDependentTestCase):
    def setUp(self):
        super(Store_Put, self).setUp()
        self.store = Store()

    def test_putOne(self):
        id = self.store.put(1, "one")
        self.assertEqual(1, self.store.get("one"))

    def test_put_twice(self):
        id = self.store.put(2, "two")
        id2 = self.store.put(2, "two")
        self.assertRaises(PlasmaObjectExists)

    def tearDown(self):
        super(Store_Put, self).tearDown()


class Store_PutGet(StoreDependentTestCase):
    def setUp(self):
        super(Store_PutGet, self).setUp()
        self.store = Store()

    def test_getOne(self):
        id = self.store.put(1, "one")
        id2 = self.store.put(2, "two")
        self.assertEqual(1, self.store.get("one"))
        self.assertEqual(id, self.store.stored["one"])

    def test_get_nonexistent(self):
        # Handle exception thrown
        with self.assertRaises(Exception) as cm:
            self.store.get("three")

        # Check that the exception thrown is a CannotGetObjectError
        self.assertEqual(cm.exception.name, "CannotGetObjectError")

    def tearDown(self):
        super(Store_PutGet, self).tearDown()


"""class Store_Notify(StoreDependentTestCase):
    def setUp(self):
        super(Store_Notify, self).setUp()
        self.store = Store()

    # Add test body here
    def test_notify(self):
        # TODO: not unit testable?


    def tearDown(self):
        super(Store_Notify, self).tearDown()"""


class Store_UpdateStored(StoreDependentTestCase):
    def setUp(self):
        super(Store_UpdateStored, self).setUp()
        self.store = Store()

    # Accessing self.store.stored directly to test getStored separately
    def test_updateGet(self):
        self.store.put(1, "one")
        self.store.updateStored("one", 3)
        self.assertEqual(3, self.store.stored["one"])

    def tearDown(self):
        super(Store_UpdateStored, self).tearDown()


class Store_GetStored(StoreDependentTestCase):
    def setUp(self):
        super(Store_GetStored, self).setUp()
        self.store = Store()

    def test_getStoredEmpty(self):
        self.assertFalse(self.store.getStored())

    def test_putGetStored(self):
        self.store.put(1, "one")
        self.assertEqual(
            1, self.store.getID(self.store.getStored()["one"])
        )  # returns ID

    def tearDown(self):
        super(Store_GetStored, self).tearDown()


class Store_internalPutGet(StoreDependentTestCase):
    def setUp(self):
        super(Store_internalPutGet, self).setUp()
        self.store = Store()

    def test_put(self):
        id = self.store.random_ObjectID(1)
        self.store._put(1, id[0])
        self.assertEqual(1, self.store.client.get(id[0]))

    def test_get(self):
        id = self.store.put(1, "one")
        self.store.updateStored("one", id)
        self.assertEqual(self.store._get("one"), 1)

    def test__getNonexistent(self):
        # Handle exception thrown
        with self.assertRaises(Exception) as cm:
            # Check that the exception thrown is a ObjectNotFoundError
            self.store._get("three")
            self.assertEqual(cm.exception.name, "ObjectNotFoundError")
            self.assertEqual(
                cm.exception.message, 'Cannnot find object with ID/name "three"'
            )

    def tearDown(self):
        super(Store_internalPutGet, self).tearDown()


class Store_saveConfig(StoreDependentTestCase):
    def setUp(self):
        super(Store_saveConfig, self).setUp()
        self.store = Store()

    def test_config(self):
        fileName = "data/config_dump"
        id = self.store.put(1, "one")
        id2 = self.store.put(2, "two")
        config_ids = [id, id2]
        self.store.saveConfig(config_ids)
        with open(fileName, "rb") as output:
            self.assertEqual(pickle.load(output), [1, 2])

    def tearDown(self):
        super(Store_saveConfig, self).tearDown()


# Test out CSC matrix format after updating to arrow 0.14.0
class Store_sparseMatrix(StoreDependentTestCase):
    def setUp(self):
        super(Store_sparseMatrix, self).setUp()
        self.store = Store()

    def test_csc(self):
        csc = csc_matrix((3, 4), dtype=np.int8)
        self.store.put(csc, "csc")
        self.assertTrue(np.allclose(self.store.get("csc").toarray(), csc.toarray()))

    def tearDown(self):
        super(Store_sparseMatrix, self).tearDown()
