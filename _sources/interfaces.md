## Flexible pipeline for neural experimentation
Here we describe a somewhat generic implementation of our software using a common 2-photon calcium imaging pipeline. In all modules, only a few things must be implemented in a specific way. Generally speaking, the user is completely free to customize the data types, control flow, specific information to be passed from module to module, etc. 

Our pipeline roughly uses the control flow:
     Acquirer --> Processor --> Analysis --> Visual


## Modules
Abstract class and template for how a module is integrated with Nexus. It contains methods to establish the in-memory data store (DS) interface from Nexus, as well as the four default queues used for signaling and data information passing. 

All modules must implement:
- setup()
- run()

In addition, the run function must check the module's internal signals queue (q_sig) for master signals from Nexus before continuing any operations. This can be done asynchronously if desired, or synchronously before each run internal step. See the Module.run() function for an example. 

The links between queues (q_in, q_out, optional extras) are defined in a configuration file. For example:

```connections:
  Acquirer.q_out: [Processor.q_in, Visual.raw_frame_queue] #second is user-defined, added as extra Link
  Processor.q_out: [Analysis.q_in]
  Analysis.q_out: [Visual.q_in]
  InputStim.q_out: [Analysis.input_stim_queue]
```
Thus each module's output queue is tied to one or more input-type queues of other modules. Nexus handles automatically creating and setting these links among modules. The user need only to put/get to/from these named queues. 

Note that each input queue must only have one source. All queues are single source, multiple drain. 


### Acquirer
Expected behavior: Continuously acquire frames while storing each frame in the DS.

The acquired frames are stored one at a time in the DS and the ID with associated frame number is passed to the output data queue (q_out). In this way, modules down the pipeline ('future') can retrieve the correct frames. Typically the data frame is here converted to the type all future modules expect, if needed. Our implementation uses 32-bit numpy arrays. 

#### BehaviorAcquirer
Expected behavior: Similar to the base Acquirer, it acquires behavioral stimulus information and syncs this information with the acquired frame.

Our implementation simply sends a number (pre-associated with a specific stimulus in setup) whenever the experiment changes stimulus directly to its output queue (q_out). An alternative method could easily place this information in the DS and use the q_out to signal a new stimulus and its ID to find it in the DS. This could be helpful if, for instance, many stimuli with fast switching are being used and the future analysis method is slower.


### Processor
Expected behavior: Continuously process frames from the DS and store the processed frames and associated estimated activity in the DS. Typically each frame is processed in sequence. However if frames are missing from the DS, drop the frame and continue with the next.

Each frame is obtained from the DS by its ID, which is obtained from the Acquirer by the input data queue (q_in). Processed frames and other data should likewise be stored by some string value and DS ID. A list of these should be passed along in q_out to the next modules.

Our implementation uses Caiman to begin processing the frames. We store the processed frame, estimated time traces, and estimated spatial traces for future modules, and pass their IDs in a list of dict pairs to future modules. 

### Analysis
Expected behavior: Continuously process frames and estimates from the DS and update new frames or estimates into the DS.

Our implementation uses the estimates and frames from the Processor (Caiman) as well as information from the input stimulus module (BehaviorAcquirer) to:
- Construct annotated frames and selected activity plots for Visual display
- Calculate tuning curves and present a subset of them for Visual display

Our default input queue (q_in) is linked to the Processor, therefore we designate a separate queue (input_stim_queue) for the input stimulus information and link it to (q_out) from the BehaviorAcquirer. The reverse configuration would also be acceptable, and is left flexible for adding other unique modules. All methods of linking queues for data information flow are easily user-defined in the configuration file.

Furthermore, our implementation of Analysis will update whenever _either_ past module has new information in the DS (as sent over the queues). It is also possible to _require_ both sets of information be present before analyzing the data by a simple cross-check of the associated frame numbers. 

### Visual
Expected behavior: Continuously update plots of imaging data and analysis. Also continuously monitor user input to send communications to Nexus. Visual is directly integrated with a FrontEnd that acts as the GUI.

All modules have access to a communications link back to Nexus (q_comm) for the purpose of, eg, indicated local state of that module, but are not required to use it. Visual is required to use this so that user commands from the GUI are passed back to Nexus. Common signals are provided in the modules.Spike class (eg, Spike.run()) and are passed directly through q_comm. 

