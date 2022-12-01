Welcome to the improv wiki!

## Table of Contents

 - [Project Overview](overview)
 - [System Design](design)
 - [API](interfaces)
 - [Milestones](milestones)
 - [System Operation](operation)

## Module Documentations
 - [Acquirer](Data Acquisition)

## File System Structure

Plan for organizing the repo:

 - `data/`: Symbolic links to synced folders (such as Dropbox) with large data files
 - `doc/`: Text-based documentation not already added into Python docstrings 
 - `src/`: Top-level directory for all source code. Further divided into modules, e.g. `src/visual/`
 - `test/`: Top-level directory for all test code. Mirrors the src directory structure

## Overview

### User Pipeline
 - Configure experimental setup: 
   * Choose image acquisition source, which chooses an Acquisition module
   * Choose processing sources, currently only caiman. Can be broken out: 
       + Motion correction
       + Neuron/component identification
       + Spike extraction
   * Choose analysis sources to obtain inferences
   * Configure other experimental parameters (size, rate, expected components, etc)
   * Configure display options: either choose module, or configure default visualization

 - Initialize server:
   * Load data store -- need to consider persistence between days and sessions
   * Load chosen experimental parameters
   * Initialize chosen pipeline components

 - Run server:
   * Pipeline components wait or run depending on signals from nexus server
   * Front end interfaces with the user and the server



### Code Pipeline
 - Front end as GUI is run first
   * Creates Nexus and communicates with it
   * Nexus can alternatively be run as stand-alone on command line
   * Has store client from Nexus from which to pull images, plots
   * Also hands off user-input in the form of configuration to Nexus
 - Nexus runs inside front-end and handles all communication 
   * Starts data store server and handles clients interfacing with the store
   * between front-end (user) and other components
   * between components in the form of signaling
 - Other components: analysis, processing, etc
   * Have access to the store through the client provided by Nexus