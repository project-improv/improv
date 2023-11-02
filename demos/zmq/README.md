# ZMQ Demo

This demo is intended to show how actors can communicate using ZMQ. There are two options for the demo, each corresponding to a different config file. The difference between these two demos is in how they send and receive messages. One uses the publish-subscribe pattern, while the other uses the request-reply pattern. 

## Instructions

Just like any other demo, you can run `improv run <path to zmq config>` in order to run the demo. For example, from within the `improv/demos/zmq` directory, running `improv run zmq_rr_demo.yaml` will run the zmq_rr demo. 

Upon running the demo, a TUI (text user interface) should show up. From here, type `setup` followed by `run` to run the demo. After it is done, type `stop` to pause, or `quit` to exit.

## Expected Output

Currently, only logging output is supported. There will be no live output during the run.

