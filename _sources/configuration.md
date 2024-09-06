(page:configuration)=
# Configuring _improv_

## Redis Configuration Options

Although _improv_ works with Redis out of the box, there are a handful of optional configuration options that can be used to change the behavior of _improv_'s data store.
Redis configuration is specified in the configuration file under the `redis_config` key.

(#persistence)=
### Persistence
_improv_ can configure the data store to write its contents to a set of append-only log files on disk.
Append-only files, or AOFs, can be reused between instances of _improv_, which will allow successive runs of _improv_ to access previously stored data.
By default, this capability is disabled, but is controlled by the following settings.

#### enable_saving
This field enables saving of data store contents to AOFs on disk. The default mode of operation uses the default Redis AOF directory, `appendonlydir`, which will be automatically created if it does not exist.
- Type: boolean (True/False)
- Example: True

#### aof_dirname
This setting controls the name of the directory in which the store saves the append-only logs.
Persistence happens automatically at scheduled intervals throughout the operation of the data store, as well as during shutdown of the data store.

- Type: string
- Example: `custom_aof_directory`

#### generate_ephemeral_aof_dirname
This field indicates that _improv_ should generate a unique AOF directory name to use each time it is run.
This will set each run of _improv_ to start from a clean data store without needing to set a unique directory name manually before each run.

- Type: boolean (True/False)
- Example: True

#### fsync_frequency
Although the append-only log files, when enabled, are updated every time the state of the data store is changed, the contents of the file are not always immediately flushed to the system's disk.
This field controls how often the data store should request that the operating system flush the contents of the append-only log files to the hard disk.

- Type: string
- Values:
  - `every_write` - sync the AOF to disk on every data store modification.
    - Highest fault tolerance
    - Highest background disk and data store load. The data store can spend a lot of time doing processing unrelated to running the experiment.
  - `every_second` - sync the AOF to disk every second.
    - Moderate fault tolerance - can only lose up to roughly 1 second of data in a disaster.
    - Better performance.
  - `no_schedule` - sync the AOF according the default system policy.
    - Lowest fault tolerance - can go from 1 to 30 seconds, occasionally more, between syncs.
    - Highest performance.
- Default: `no_schedule`
- Example: every_second

### Connectivity
#### port
This field controls the port on which the data store runs. If left unspecified, _improv_ will attempt to start the data store on the default port number,
looking for alternatives if tried port numbers are busy. If specified, then _improv_ will only try to start the data store on the specified port number.

- Type: integer
- Default: 6379
- Example: 16300

### Example
An _improv_ configuration file which sets the data store port to 6385, enables persistence to disk, and uses a unique directory for each run, would look like:
```
actors: ...

connections: ...

redis_config:
  enable_saving: True
  port: 6385
  generate_ephemeral_aof_dirname: True
```