## [Master Server](server)
 - Purpose: To handle the entire pipeline and communication between other classes in the system. Also manages disk I/O for now.
 - Not persistent. Created during system initialization based on user input or config files.
 - Class name: NeuralNexus

### Attributes
 - All other objects in the processing pipeline
 - Data store: includes frames, logs
### Methods
 - For data store:
    - _startStore: start running plasma server in subprocess
    - _closeStore: kill subprocess running server
 - Load objects: Based on user selection for acquisition, processing, etc
 - Run: Multiprocessing of simultaneous/asynchronous running objects
 - Update visualization
### Test cases
 - Unit and system integration tests


## [Front Panel](front-panel)
 - Purpose: To handle all visualization and user input. Communicates with the server. 

### Attributes
 - UI front panel
 - ConfigParam
### Methods
 - Update handler (UI)
 - Plot current frame
 - Plot components
 - Plot activity
### Test cases


## [Data Store](store)
 - Purpose: Backend data store. Handled by server.
 - Current implementation in Pyarrow Plasma. Class name Limbo.
 - Can be implemented with different store types.

### Attributes
 - PlasmaClient to communicate with store
 - dict to hold internal correspondence between simple object names and their IDs  -- limited use
### Methods
 - connectStore: create a client to connect to store
 - get: get items from store
 - put: put items into store


## [Configuration Parameters](config-params)
 - Purpose: To contain any and all configuration parameters for the experiment.
 - Logs (via the server) initial and changed states
 - Class also wraps CNMFParams

### Attributes
 - CNMFParams
 - List of objects selected for the experiement
 - Timestamp
### Methods
 - Update: configuration change
 - Save: to server to write to disk as log
### Test cases


## [Image Acquisition](acquisition)
 - Purpose: Acquire an image from a source to hand off to the server.
 - Implements some kind of buffer to ensure no lost frames.
 - Class wraps imagemp functionality
 - Abstracted to allow multiple classes depending on how images/frames are acquired.

### Attributes
 - Buffer: holding certain number of frames
 - Current frame
### Methods
 - Acquire_image: get next frame
 - Save: hand off to server
### Test cases


## [Image Processing](process)
 - Purpose: Given an image (from file or shared structure), produce neural activity information.
 - Should include motion stabilization, neuron-finding, and spike identification.
 - Class wraps CaImAn functionality (OnACID, CNMF)
 - Is also abstracted

### Attributes
 - Current frame: frame being analyzed
### Methods
 - Get_next_frame: from server
 - Process_frame: analyze current frame
 - Save: hand off to server
### Test cases


## [Activity](activity)
 - Purpose: To contain the object describing neural activity
 - Class name: NeuralExciton
 - Class wraps CaImAn objects (Estimates)

### Attributes
 - Estimates
### Methods
 - Normalize components
 - Filter components: take a subset, threshold, etc
 - Compare components
### Test cases


## [Inference](inference)
 - Purpose: Given some neural activity, produce an online estimate of the effects of each neuron's response. 
 - Class name: Neurlock

### Attributes
 - Chosen method for analysis
### Methods 
 - Evaluate components
 - Compute residuals
### Test cases


## [Stimulation](stimulation)
 - Purpose: To send a stimulus to the subject under study.
 - Class: not yet implemented