from test.plasmaTest_py import startStore
from improv.store import Store

import multiprocessing as mp

import pyarrow as pa

from scipy.sparse import csc_matrix

# For logging? Already included in Store class?
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Check if error occurs if connectStore before terminal command to start Plasma store?

# Instantiate Store using all default attributes.
store = Store()

# Test connectStore method
client = store.client

# Terminal command to start Plasma store.
p1 = mp.Process(target=startStore)
p1.start()
p1.join()

# Instantiate Store using all default attributes.
store = Store()

# Test connectStore method.
client = store.client


# Test put method.
# pa.array() handles 1D arrays
# pa.Tensor.from_numpy() handles 2D+ arrays
def test(data, object_name, use_pyarrow=False):
    store = Store()

    if use_pyarrow:
        if data.ndim > 1:
            obj = pa.array(data)
        else:
            obj = pa.Tensor.from_numpy(data)
    else:
        obj = data

    objID1 = store.put(obj, object_name)
    # Why??? Does this do the same??? Seems redundant/odd -> output of client.put() is objID, how could the input also be objID?
    objID2 = store._put(obj, objID1)

    # If creating buffer and sealing...
    # Necessary?
    # objSize = data.nbytes

    # client.create_and_seal(objID, obj)

    # What was stored? Check data was stored.
    # store.getStored()
    #   output: stored (dict)
    # get():
    #   input: object_name (str) from stored (dict)
    #   output: _get out
    # getID():
    #   input: objID (str)...why from stored (dict) by object_name?
    #   output: obj (data...array/tensor, etc.)
    # _get():
    #   input: obj_name (str)
    #   * uses getID()
    #   output:

    return (
        store.getStored(),
        store.get(object_name),
        store.getID(objID1),
        store._get(object_name),
    )


# Test two at same time - meaning, put two objects in at the same time, or use two clients at the same time?

# Test min (including nulls) and max (2D and tensors).
# min - all zeros.
data = np.empty(0)
objName = "empty"
stored, data1, data2, data3 = test(data, objName)
print(
    stored,
    np.array_equal(data1, data2),
    np.array_equal(data1, data3),
    np.array_equal(data2, data3),
)

data = np.identity(100)
objName = "identity"
stored, data1, data2, data3 = test(data, objName)
print(
    stored,
    np.array_equal(data1, data2),
    np.array_equal(data1, data3),
    np.array_equal(data2, data3),
)

data = csc_matrix((100, 100), dtype="int64").toarray()
objName = "csc_matrix"
stored, data1, data2, data3 = test(data, objName)
print(
    stored,
    np.array_equal(data1, data2),
    np.array_equal(data1, data3),
    np.array_equal(data2, data3),
)
