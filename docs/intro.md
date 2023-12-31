# _improv_: A platform for adaptive neuroscience experiments


_improv_ is a lightweight package designed to enable complex adaptive experiments in neuroscience. This includes both traditional closed-loop experiments, in which low latency and real-time analysis are the priority, and experiments that attempt to maximize neural responses by changing the stimulus presented. In each case, the design of the experiment must change online based on previously seen data.

In practice, such experiments are technically challenging, since they require coordinating hardware with software, data acquisition and preprocessing with analysis. We build _improv_ to provide a simple means of tackling these coordination problems. In _improv_, you specify your experiment as a data flow graph, write simple Python classes for each step, and we handle the rest. 

:::{figure-md} markdown-fig
![](https://dibs-web01.vm.duke.edu/pearson/assets/improv/improvGif.gif)

Raw two-photon calcium imaging data in zebrafish (left), with cells detected in real time by [CaImAn](https://github.com/flatironinstitute/CaImAn) (right). Neurons have been colored by directional tuning curves and functional connectivity (lines) estimated online, during a live experiment. Here only a few minutes of data have been acquired, and neurons are colored by their strongest response to visual simuli shown so far.
:::

## Why _improv_?

### Skip the messy parts
Need to collect data from multiple sources? Run processes on different machines? Pipeline data to multiple downstream analyses? _improv_ handles the details of communication between processes and machines so you can focus on what's important: defining the logic of your experiment. 

### Simple design
_improv_'s design is based on a simplified version of the Actor Model of concurrency {cite:p}`agha1986actors`: the experiment is a directed graph, the nodes are actors, and data flows by asynchronous message passing. Apart from this, we strive for maximum flexibility. _improv_ allows for arbitrary Python code in actors, allowing you to interoperate with the widest variety of tools. Examples can be found in [](page:demos) and our [`improv-sketches`](https://github.com/project-improv/improv-sketches) repository.

![](https://dibs-web01.vm.duke.edu/pearson/assets/improv/actor_model.png)

### Speed
_improv_ is designed to be fast enough for real-time experiments that need millisecond-scale latencies. By minimizing data passing and making use of in-memory data stores, we find that many applications are limited by network latency, _not_ processing time.

### Reliability
_improv_ is designed from the ground up to be fault-tolerant, with a priority on data integrity. Individual processes may crash, but the system will keep running, ensuring that a single bad line of code does not result in data loss. In addition, we provide the tools to produce an audit trail of everything that happened online, ensuring you don't forget that one setting you changed mid-experiment.

<!-- ```{bibliography}
``` -->