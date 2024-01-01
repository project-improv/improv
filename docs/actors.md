(page:actors)=
# Writing actors

## Overview
Nearly any experiment implemented in _improv_ will require the creation of custom actors. As explained in [][page:design:pipeline_spec], these actors are defined by Python classes and represent distinct, independent steps in the processing pipeline. The best way to learn to write actors is by example: the [demos](https://github.com/project-improv/improv/tree/main/demos) give several complete, runnable pipelines.

Here, we explain some of the theory behind how actors work, as well as some more advanced functionality.

## Basic expectations: the `AbstractActor` class

The `AbstractActor` class defines the template from which all actors must inherit. You will never directly use an `AbstractActor`, but all other actors are based on it. You can look up the documentation in the [](autoapi/index), but most of the methods associated with the class are internal, handling communication with the store and server, and you won't need to deal directly with them. The three methods work knowing about are:
- `setup`: handles all the setup work the class needs to do _before_ the experiment starts
- `run`: the data processing step executed repeatedly during the experiment; handles getting input data from the store, putting output data into the store, and informing other actors where the results are
- `stop`: handles all cleanup necessary when the class is stopped

However, the basic `AbstractActor` lacks many features we might want: it doesn't properly handle signals from the server (including `stop`!) and so requires writing a lot of tedious code to check for other things that might be happening in the system. As a result, _improv_ provides the `ManagedActor` class, to which we turn next.

## Practical actor implementations with `ManagedActor`
The key benefit the `ManagedActor` class (which is aliased to `Actor`) offers over the more general `AbstractActor` is the addition of the `RunManager` context manager. This context manager is used within `run` and handles communication with the server, including calling the `setup` and `stop` methods when the actor receives those signals. In a `ManagedActor`, the actual processing logic is located in the `runStep` function, which must be defined for any valid actor subclass.[^async_note] Again, examples of actors subclassing `Actor` (aka, `ManagedActor`) are available in the `actors` subfolder of each demo in [demos](https://github.com/project-improv/improv/tree/main/demos).

Internally, actors communicate with each other and with the server via [multiprocessing queues](https://docs.python.org/3/library/multiprocessing.html#pipes-and-queues), which are highly performant but restricted to processes on the same machine. For actors located on other machines, or across networks, there are [other actors](https://github.com/project-improv/improv/blob/main/demos/sample_actors/zmqActor.py) that communicate using [ZMQ](https://zeromq.org)[^zmq_note]. In any event, the details should be transparent to users, and implementations are subject to change without notice, so users should not depend on these internals.

[^async_note]: In addition, there are asynchronous versions of the `ManagedActor` and `RunManager`, and these may become the defaults aliased to `Actor` in future versions, so users should not rely on details of these implementations.
[^zmq_note]: And this option may become the default in future versions.
