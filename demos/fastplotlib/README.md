# fastplotlib demo

This demo consists of a **generator** `actor` that generates random frames of size 512 * 512 that are sent via a queue to a **processor** `actor` that can be used to process the frames and send them via `zmq`. The `fastplotlib.ipynb` notebook then receives the most recent frame via `zmq` and displays it using [`fastplotlib`](https://github.com/kushalkolar/fastplotlib/).

Usage:

```bash
# cd to this dir
cd .../improv/demos/fastplotlib

# start improv
improv run ./fastplotlib.yaml

# call `setup` in the improv TUI
setup

# Run the cells in the jupyter notebook until you receive
# the dark blue square in the plot

# once the plot is ready call `run` in the improv TUI
run
```
