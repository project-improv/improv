# Introduction

For demonstration purposes, we have provided an example of an actor system within improv. This example can be found in the `improv/pytest` folder and is composed of 4 separate files: sample_generator.py, sample_processor.py, sample_demo.py, and sample_config.py. The first two can be found in the `pytest/actors` directory while the last can be found in the `pytest/configs` directory. 

## Overview

Recall that improv is a software platform that is composed of "actors" that communicate with each other. The arrangement of these actors and the communication between them can be modeled as a directed acyclic graph, where each actor is a node, and each edge is a communication link. Recall also that there is a server class called "Nexus" that handles the actors, as well as a data store called "Store". 

In this example, we will have two actors, denoted "Generator" and "Processor", each connected to "Store". We have the following diagram:

![improv_diagram](https://user-images.githubusercontent.com/104780909/199650148-7e9493c5-1ba6-4f28-9e03-283d5a28efc1.jpg)

At each time step, Generator will generate 5 integers from 1-100, and send them to Processor. Processor will then compute the average of these 5 numbers and output them to stdout. It will also keep track of a cumulative average. 

\*\*\*Note that the arrows between the store and the actors is dashed, unlike the arrow between actors. This is to denote that while the link between actors is an object(an instance of Link.py) that explicitly allows communication, any actor by default has read and write access to the store. 

# Running the demo

To start the demo, execute the following commands:
1. Navigate to the `improv/pytest` directory in the command line. 
2. Run `python sample_demo.py`. 
3. When you see that improv has started all processes, type `setup` into the command line. This will begin the setup process, executing user defined setup functions. You should see the following output:

![improvwikioutput1](https://user-images.githubusercontent.com/104780909/199650389-dd0da8c4-7134-4c43-9934-d0cb1b00c72b.png)

4. Once improv has allowed start, type `run` into the command line. This will start running each actor, executing user defined run functions.
5. To end the demo, type `quit` into the command line. 

# Explanation

For an explanation of what is going on internally when improv is running, please consult the [wiki page](https://github.com/project-improv/improv/wiki/Building-your-own-actor-system) on building your own actors and actor system. For now, we can give a high level explanation.

## Setup

When "setup" is run, every actor is sent a signal to execute their setup function. This can be initializing class variables, loading data, or anything else that is imperative for the actor to have before running. In our example, Generator first initializes the first 5 sets of 5 integers to be sent to processor. Processor's setup function initializes its attributes such as its name. **Every actor *must* have a setup function.**

## Run

When "run" is run, every actor is sent a signal to execute their run function. This can be a wide range of things, from acquiring data to displaying a GUI. The run function consists of a RunManager, a class used to allow the run to be modified during runtime. Along with this, an actor must also specify a helper function to be passed into the RunManager as an argument. The helper function contains the bulk of the functionality for run. For Generator, this helper function consists of generating 5 integers and passing it into its communication link to the store. For Processor, the run helper function gets the integers from Generator, computes the average and outputs it to stdout. **Every actor *must* have a run function.**


## The config file

The config file specifies the identity and graph structure of the actors. For more of an explanation, please consult the [wiki page](https://github.com/project-improv/improv/wiki/Building-your-own-actor-system) on building an actor system.

## The role of Nexus 

Nexus is a server class that handles all the backend for the actors; this includes, among many other things, processing user input (like what happened when you type `setup` into the terminal), constructing the actor system from the config file, and initializing the store. It is **not** an actor. 
