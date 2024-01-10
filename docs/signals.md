(page:signals)=
# Signals and communicating between actors
In [](page:actors), we covered the basics of writing your own `Actor` class. Here, we consider a further complication: 
How do actors talk with _improv_ and with one another. The most basic mechanism for doing so is through _signals_.

On Unix-like computing systems, signals are messages sent by the operating system to processes running on the machine.
The most well-known of these are `SIGINT` ("interrupt," what you get by pressing `Ctrl-C` at the terminal), 
`SIGTERM` ("terminate," shut the process down, but ask nicely), and `SIGKILL` ("kill," and don't take no for an 
answer). In Python, handling of these events is done with the [`signal`](https://docs.python.org/3/library/signal.html) 
library.

For communicating between _improv_'s server and actors, we use both the `signal` library (internally) and 
`actor.Signal` for _improv_'s own set of messages. In the default `Actor` class (an alias for `ManagedActor`), 
signals are handled by the `RunManger` class, which calls the relevant class method when it receives a signal. 
In [](tables:signals) we list the signals defined in `Signal` along with the `Actor` methods they call.

```{table} Correspondences between signals
:name: tables:signals

| `actor.Signal` | `ManagedActor` method called |
|---|---|
| `setup` | `setup` |
| `run` | `runStep` |
| `pause` | not yet implemented |
| `resume` | not yet implemented |
| `reset` | not yet implemented |
| `load` | not yet implemented |
| `ready` | sent by actors to server |
| `kill` | handled by server |
| `revive` | not yet implemented |
| `stop` | `stop` |
| `stop_success` | not yet implemented |
| `quit` | handled by server |
```

```{note}
Users who wish to do their own signal handling (e.g., by inheriting from `AbstractActor`) will need to test for
the presence of `actor.Signal` messages in the connection to the server within the `run` function and handle 
them appropriately.
```