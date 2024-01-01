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

Pipelines in _improv_ are specified in [YAML files](https://yaml.org). In general, _improv_ configuration files contain three top-level headings:
1. `Settings` includes program settings to be passed to the server upon startup.