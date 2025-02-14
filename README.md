
<img src="docs/improv_logo_horizontal.svg" style="width:40%"/>

Adaptive experiments for neuroscience
---------

[![PyPI](https://img.shields.io/pypi/v/improv?style=flat-square?style=flat-square)](https://pypi.org/project/improv)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/improv?style=flat-square)](https://pypi.org/project/improv)
[![docs](https://github.com/project-improv/improv/actions/workflows/docs.yaml/badge.svg?style=flat-square)](https://project-improv.github.io/)
[![tests](https://github.com/project-improv/improv/actions/workflows/CI.yaml/badge.svg?style=flat-square)](https://project-improv.github.io/)
[![Coverage Status](https://coveralls.io/repos/github/project-improv/improv/badge.svg?branch=main)](https://coveralls.io/github/project-improv/improv?branch=main)
[![PyPI - License](https://img.shields.io/pypi/l/improv?style=flat-square)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/psf/black)


_improv_ is a streaming software platform designed to enable adaptive experiments. By analyzing data as they arrive, we can obtain information about the current brain state in real time and use it to adaptively modify an experiment as data collection is ongoing. 

![](https://dibs.duke.edu/sites/default/files/improvGif.gif)

This video shows raw 2-photon calcium imaging data in zebrafish, with cells detected in real time by [CaImAn](https://github.com/flatironinstitute/CaImAn), and directional tuning curves (shown as colored neurons) and functional connectivity (lines) estimated online, during a live experiment. Here only a few minutes of data have been acquired, and neurons are colored by their strongest response to visual simuli shown so far.
We also provide up-to-the-moment estimates of the functional connectivity by fitting linear-nonlinear-Poisson models online, as each new piece of data is acquired. Simple visualizations offer real-time insights, allowing for adaptive experiments that change in response to the current state of the brain.


### How improv works

<img src="https://dibs.duke.edu/sites/default/files/improv_design.png" width=85%>

improv allows users to flexibly specify and manage adaptive experiments to integrate data collection, preprocessing, visualization, and user-defined analytics. All kinds of behavioral, neural, or modeling data can be incorporated, and input and output data streams are managed independently and asynchronously. With this design, streaming analyses and real-time interventions can be easily integrated into various experimental setups. improv manages the backend engineering of data flow and task execution for all steps in an experimental pipeline in real time, without requiring user oversight. Users need only define their particular processing pipeline with simple text files and are free to define their own streaming analyses via Python classes, allowing for rapid prototyping of adaptive experiments.  
  <br />
  <br />
  
<img src="https://dibs.duke.edu/sites/default/files/actor_model.png" width=60%>

_improv_'s design is based on a streamlined version of the actor model for concurrent computation. Each component of the system (experimental pipeline) is considered an 'actor' and has a unique role. They interact via message passing, without the need for a central broker. Actors are implemented as user-defined classes that inherit from _improv_'s `Actor` class, which supplies all queues for message passing and orchestrates process execution and error handling. Messages between actors are composed of keys that correspond to items in a shared, in-memory data store. This both minimizes communication overhead and data copying between processes. 



## Installation

For installation instructions, please consult the [documentation](https://project-improv.github.io/improv/installation.html).


### Contact
To get in touch, feel free to reach out on Twitter <a href="http://twitter.com/annedraelos" target="_blank">@annedraelos</a> or <a href="http://twitter.com/jmxpearson" target="_blank">@jmxpearson</a>. 
