import os
import uuid
import pytest

store_loc = str(os.path.join("/tmp/", str(uuid.uuid4())))


@pytest.fixture
def set_store_loc():
    return store_loc
