import pytest
from improv.store import *

# https://docs.python.org/3/library/pickle.html#what-can-be-pickled-and-unpickled
# Can be pickled:
# None, True, and False;
# integers, floating-point numbers, complex numbers;
# strings, bytes, bytearrays;
# tuples, lists, sets, and dictionaries containing only picklable objects;
# functions (built-in and user-defined) defined at the top level of a module (using def, not lambda);
# classes defined at the top level of a module;
# instances of such classes whose __dict__ or the result of calling __getstate__() is picklable (see section Pickling Class Instances for details).

# https://quick-adviser.com/what-cannot-be-pickled-in-python/
# Cannot be pickled:
# Classes, functions, and methods - if you pickle an object, the object’s class is not pickled, just a string that identifies what class it belongs to. 
# With pickle protocol v1, you cannot pickle open file objects, network connections, or database connections.

# Test values for connectStore:
# store_loc will never be None because it is initialized w/class?
TEST_store_loc = None

# Test values for put/get:
TEST_1 = None
TEST_2 = True
TEST_3 = False

# Fixtures
@pytest.fixture
def store():
    return Limbo()

def test_connectStore(store):
    store = Limbo.connectStore()
    # What would self.client be? How to assert truth?
    assert store == 

# ???
@pytest.mark.parameterize("object")
def test_put_get(store):
    obj_put = Limbo.put(object)
    obj_get = Limbo.get(obj_put)
    