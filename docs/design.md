(page:design)=
# _improv_'s design

Here, we cover some technical details that can be useful in understanding how _improv_ works under the hood. 

One way of viewing _improv_ is as a lightweight networking library that handles three key tasks:
1. **Pipeline specification and setup.** Experimental pipelines are defined by YAML files. At run time, _improv_ reads these files and starts separate system processes corresponding to each processing step (actor).
1. **Interprocess communication.** As part of each experiment, data need to be collected and passed between processes. _improv_ uses a centralized server and in-memory data store to coordinate this process.
1. **Logging and data persistence.** After the experiment is done, we need records of both what happened during the experiment and what data were collected. _improv_ organizes both of these processes by creating a unified log file and storing data to disk as the experiment runs.

We consider each of these components below.

(page:design:pipeline_spec)=
## Pipeline specification

_improv_ pipelines are [directed graphs](https://en.wikipedia.org/wiki/Directed_graph), with each node corresponding to a processing step (called an "actor" after the [actor model](https://en.wikipedia.org/wiki/Actor_model) of concurrency {cite:p}`agha1986actors`) and each link corresponds to data passed between actors. For instance, a simple experiment in which calcium fluorescence images are read in from a microscope, processed via [CaImAn](https://github.com/flatironinstitute/CaImAn), fit using a [linear-nonlinear-Poisson model](https://en.wikipedia.org/wiki/Linear-nonlinear-Poisson_cascade_model) along with information about the stimulus presented, and displayed via a GUI might look something like

:::{figure-md} example_dag
![](https://dibs-web01.vm.duke.edu/pearson/assets/improv/example_dag.svg)

An example directed graph corresponding to an _improv_ experimental pipeline.
:::

Pipelines in _improv_ are specified by [YAML files](https://yaml.org). _improv_ configuration files contain three top-level headings:
1. `settings` includes program settings to be passed to `nexus.Nexus.createNexus` upon startup. This includes control and output port numbers to be used for input and output to the server, respectively. See the documentation for `nexus.Nexus.createNexus` for other arguments.
1. `actors` is a list of actors that form the nodes of a directed graph. Each item requires two attributes: `package` gives the name of the Python file containing the actor definition and `class` gives the name of the class within that file. As described in [](page:running:options), _improv_ will search for actors in the directory containing the YAML config file by default, though more directories can be specified with the `--actor-path` option. **Other attributes will be passed directly (as a dictionary) to the actor class constructor.**
1. `connections` is a list of connections between actors. Each item contains the name of an actor output (e.g., `Processor.q_out`) and a list of actors that will receive this output. 

The example graph of [the figure above](example_dag) is implemented in the [zebrafish demo](https://github.com/project-improv/improv/blob/main/demos/naumann/naumann_demo.yaml), whose YAML file is given by
```
actors:
  GUI:
    package: actors.visual_model
    class: DisplayVisual
    visual: Visual

  Acquirer:
    package: improv.actors.acquire
    class: TiffAcquirer
    filename: data/recent/xx.tif
    framerate: 2

  Processor:
    package: actors.processor
    class: CaimanProcessor
    init_filename: data/recent/xx.tif
    config_file: naumann_caiman_params.txt

  Visual:
    package: actors.visual_model
    class: CaimanVisual
  
  Analysis:
    package: actors.analysis_model
    class: ModelAnalysis

  InputStim:
    package: improv.actors.acquire
    class: StimAcquirer
    filename: data/recent/stim_freq.txt

connections:
  Acquirer.q_out: [Processor.q_in, Visual.raw_frame_queue]
  Processor.q_out: [Analysis.q_in]
  Analysis.q_out: [Visual.q_in]
  InputStim.q_out: [Analysis.input_stim_queue]
```
The particular details are not so important here as the logic of how pieces of the experiment are put together into a single file.

## Interprocess communication

To understand how _improv_ translates a pipeline specified in a YAML file to a working experiment, it's helpful to consider what happens after `improv run` is called:
1. An instance of the server is created using the specified configuration file and ports.
    1. The configuration file is loaded and parsed. Ports specified in the configuration file are overridden by ports specified at the command line.
    1. If no ports were specified, random available ports are chosen. One port (`control_port`) is for incoming instructions to the server (e.g., from GUI, TUI, etc.). The other port (`output_port`) is for broadcast status messages from the server.
    1. The server starts the in-memory data store (with size specified (in bytes) in the `settings` section of the YAML file).
    1. The server connects to the store and subscribes to its notifications.
    1. The server loops over actors in the configuration file, creating an instance of each class for each actor.
    1. The server loops over connections, creating a communication channel between each pair of actors.
1. The server is started.
    1. Using [`multiprocessing`](https://docs.python.org/3/library/multiprocessing.html), each actor's `run` method is launched (via either spawn or fork, as specified by the actor's `method` attribute in the YAML file) in a separate process.[^run_warning]
    1. The server starts an event loop that listens for input from either the control port or the actors. An "Awaiting input" message is sent on the output port.
    1. The server writes its port configuration to the log file, to be read by clients who wish to connect.
1. The textual user interface (TUI) client is started and connects to the server. Other clients may also connect to the server's control port and send commands.
1. At this point, clients may send any of the messages defined in `improv.actor.Signal`, including `setup`, `run`, `stop`, and `quit`. See [](page:signals) for more details.

What is also important to realize is that none of the above directly pertains to data flow. Once the links between actors are set up, each actor's `run` method is responsible for 
1. listening on its incoming links for addresses (keys) of newly available data from each parent actor in the graph
1. retrieving new data items directly from the store (by key)
1. performing whatever processing is required
1. depositing its outputs in the store
1. broadcasting the address(es) of its data outputs to its children in the graph

For examples and further documentation, see [](page:actors).

[^run_warning]: Note that users will _not_, in most cases, overload the `run` command for actors. They should instead write the `runStep` function. See [](page:actors).

## Logging and persistence
Finally, _improv_ handles centralized logging via the [`logging`](https://docs.python.org/3/library/logging.html) module, which listens for messages on a global logging port. These messages are written to the experimental log file. 

Data from the server are persisted to disk using [Redis Append-Only Log Files](https://redis.io/docs/latest/operate/oss_and_stack/management/persistence/) (if `redis_config: enable_saving` is set to `True` in the configuration file). See the [persistence section](configuration.md#persistence) of the [configuration guide](configuration.md) for more information.