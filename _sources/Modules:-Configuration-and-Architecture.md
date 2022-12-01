## Overview
`Nexus` is the central controller of `improv`. It performs the following:
1. Start all modules in its own process (based on `Tweak`).
2. Creates bidirectional communication links between itself and each module.

## Configuration File
Nexus configuration is stored in the `Tweak` class, which is parsed from a YAML file (specified at `createNexus`). The file must be structured as in the example below.

```yaml
modules:
  Acquirer:
    package: acquire.acquire
    class: FileAcquirer
    filename: ../data/Tolias_mesoscope_2.hdf5
    framerate: 15

  Processor:
    package: process.process
    class: CaimanProcessor
    config_file: ../caiman_config.yml

connections:
  Acquirer.q_out: [Processor.q_in]
```
Here, there are two modules named `Acquirer` and `Processor`. `package` and `class` are required. `class` refers to the class name of the module and `package` refers to the package that contains that class. `config_file` is an optional keyword for a path to an extra YAML configuration file. This file is parsed automatically into a `dict` and passed into `__init__` as a `config` keyword argument. All remaining key-value pairs are passed as keyword arguments to the `__init__` method of each class. 

This is how Nexus instantiates a class based on the YAML file above.
```python
class FileAcquirer(Module):  # in src/acquire/acquire.py
    def __init__(self, filename='../data/Tolias_mesoscope_2.hdf5', framerate=15):
        pass
```

`connections` specifies communication links based on `multiprocessing.Queue`. The format is `input: output`. Note that the output needs to be a `list`, which also means that links with multiple destinations are allowed. In this case, `Acquirer` has a link to `Processor`. In addition to this, all modules have two-way links to/from `Nexus` (`q_comm` and `q_sig`, respectively).

## Tweak
The Tweak class is an internal representation of the configuration file. It is parsed from the YAML configuration file. Like the YAML, it has two main attributes: `modules` and `connections`, both of which are `dict`.

### `Tweak.modules`
A `dict` that contains the configuration of all modules in a `name: TweakModule` format. The TweakModule class signature is shown below.

```python
class TweakModule:
    self.name = name
    self.packagename = packagename
    self.classname = classname
    self.options = dict()
    self.config = dict()
```
The `package` and `class` key from the YAML is passed into `self.packagename` and `self.classname`, respectively. `self.config` is a parsed YAML file from `config_file` (if specified). All other keyword arguments are stored in `self.options`.

### `Tweak.connections`
This `dict` is the exact `dict` representation of the `connections` segment in the YAML file.

### Dynamic Tweak Update
pass

----------------------------------
## Nexus
The Nexus controller also has information about modules and connections, loaded from Tweak. 

### Modules
First, Nexus gets `self.tweak.modules` from Tweak and instantiates each item based on the TweakModule.classname, etc. It ensures each modules gets a store connection and signal and communication Links (`q_comm, q_sig`).
It then only stores the name and instance in the dictionary self.modules: `self.modules.update({name:instance})`.
It also stores the signal and communication Links in two other dictionaries `self.comm_queues` and `self.sig_queues`.

This is not to be confused with the modules dictionary in Tweak, though they use the same `name` as a key. Tweak stores configuration information (options, user input) and Nexus stores runtime information. A Tweak object can be saved for later usage; Nexus cannot.  

### Connections
For each source and drain (list) in `self.tweak.connections` is used to create new Links and assign them to the relevant modules. 
(If there are multiple drains, a MultiLink is used to replicate information from the single source to each drain.)
The module `name` (again, consistent between Tweak.modules and Nexus.modules) is used as a part of each Link name. Link.name is typically `name+'_multi'` or `name+'_'+drain_name` to uniquely identify that Link and give a clue as to its purpose. 

#### Link
```
Link(): or MultiLink():
  self.name
  self.start
  self.end
```
For identification, each Link has a name, source (start), and drain (end). In the case of a MultiLink, `self.end` is a list.

 