## Pipeline for online processing using CaImAn

### Initialization
 - Get CNMFParams or similar object from server
 - Instantiate OnACID object using CNMFParams 
 - Call OnACID.initialize_online()

### Real-time processing
Overall need to break up and wrap existing OnACID.fit_online()
 - Wait for frame from server
 - For each given frame, run local method process_frame()
   * Adjust frame based on downsampling, normalizing
   * Motion correction
   * If requested, also update shapes for neuron finding
 - Return Estimates or similar object to data store

### Benchmark
<img width="600" alt="Memory Consumption" src="https://user-images.githubusercontent.com/34997334/63104517-1fe50300-bf4d-11e9-8bd1-463d50cbd8df.png">



## Suite2p
Instead of CaImAn, Suite2p could be used to process raw frames. This is demonstrated in `batch_process.py`.

### Parameters
 - `buffer_size=200` Number of frames to accumulate before starting Suite2p.
 - `path='../output'` Path to saved TIFF files.
 - `max_workers=2` Maximum number of Suite2p instances to be running concurrently.

### Description
Since Suite2p cannot perform an online analysis, we have to save raw frames into batches of size `buffer_size`.Each batch is stored as a stack in a TIFF file as specified in `path`. Afterwards, Suite2p is called. If Suite2p has not finished processing when a new batch arrives, a new process is spawned to run another instance of Suite2p until `max_workers` is reached.