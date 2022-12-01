Offline store is implemented using the [LMDB](https://lmdb.readthedocs.io/en/release/) key-value database. This system serves as an storage extension to the online store while allowing data to be accessed after the experiment.
This is located in `nexus.store.LMDBStore`.

### Parameters
`use_hdd=False` Enable/Disable LMDB functionality

`hdd_maxstore=1e12` Maximum database size. There is no penalty in setting this as big as possible.

`hdd_path='output/'` Path to database output.

`flush_immediately=False,` Write information to disk immediately. Big performance penalty. For critical experiments.

`commit_freq=20` Get the data into a writing queue every _ frames.

### Description
When `use_hdd` is set to `True`, all objects that are put in to the `Store` is captured and saved on to the LMDB database with the following key `{originating module}{frame number}{time.time()}`. Once a module requests an object that is not available in the `PlasmaStore`, the object ID gets translated into the LMDB key, and the object is retrieved.

### Offline Access
Once the experiment is finished, the data remain available and can be analyzed using `utils.reader.LMDBReader`. The API is as followed.

```python
def __init__(self, path):
    """
    Constructor for the LMDB reader.
    :param path: Path to LMDB folder
    :type path: str
    """

def get_all_data(self):
    """
    Load all data from LMDB into a dictionary.
    Make sure that the LMDB is small enough to fit in RAM.
    :return: All data in a dictionary.
    :rtype: Dict[str: object]
    """

def get_data_types(self):
    """
    Return all data types defined as {object_name}, but without number.
    :rtype: Set[str]
    """

def get_data_by_number(self, t):
    """
    Return data at a specific frame number.
    :param t: Frame number
    :type t: int
    :return:
    :rtype: Dict[str: object]
    """

def get_data_by_type(self, t):
    """
    Return data with key that starts with {t}.
    :param t: Data prefix
    :type t: str
    :rtype: Dict[str: object]
    """

def get_params(self):
    """
    Return parameters in a dictionary.
    :rtype: Dict[str: object]
    """
```