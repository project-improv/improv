Raw image frames can be sent into the system using a HDF5 file and MATLAB.

## From HDF5
Using the demo file from CaImAn, `Tolias_mesoscope_1.h5`, the system gets the image stack in a NumPy array and sends out images every `1/framerate` second.

## From Scanbox
Scanbox offers real-time frame sharing via a memory-mapped file ([more details](https://scanbox.org/2016/02/11/memory-mapped-data-sharing-for-real-time-processing/)). However, these could only be accessed quickly within the MATLAB infrastructure. Therefore, we need to create another MATLAB instance in order to access the memory-mapped data.

To demonstrate data transfer from a memory-mapped file, we create a test system with two files: `header` and `img` (check the comments in `generator.m`). The header is an `int16` array that contains status information (ex: clear to start, new data available). The img is an `struct int16` with dimensions of a single image frame. Make sure to run `generator.m` before running `improv`.

```matlab
img = memmapfile('scanbox.mmap', 'Format', {'int16', [512, 512], 'data'}, 'Writable', true);
```
Note that the parameters that are used to load the memmapfile must exactly match the parameters of the array(s) that creates the memmapfile. Otherwise, MATLAB will throw this error.
> A subscripting operation on the Data field attempted to create a comma-separated list. The memmapfile class does not support the use of comma-separated lists when subscripting.

\
In `acquirer_matlab.py`, we start a MATLAB instance using the MATLAB Engine API for Python and create a `memmapfile` object. MATLAB commands can be executed using the `eng.eval` function. **Important: note that running any MATLAB commands that do not have a return value requires the keyword** `nargout=0`.

```python
eng = matlab.engine.start_matlab()
eng.eval('a = 2', nargout=0)
```

When the module receives the `run` signal, it changes a value in the `header` array to start data generation by `generator.m`. To receive data, the system runs a MATLAB snippet that holds the execution until the image is updated by `generator.m`. Afterwards, the new image is retrieved and converted into a NumPy array.

```python
raw: matlab.mlarray.int16 = self.eng.eval('img.Data.data')
frame = np.array(raw._data).reshape(raw.size[::-1]).T
```
### Benchmark
<p align="center"><img width="600" alt="Time per cycle of MATLAB acquisition" src="https://user-images.githubusercontent.com/34997334/63043443-40a05080-be9a-11e9-9a68-82331616b30e.png"></p>
Time per cycle of MATLAB acquisition (440x256 int16) measured in `improv`. This is the time it takes to finish one cycle of acquiring less the wait time (1/15 s).

## From ScanImage
ScanImage offers the ability user-defined functions upon a [pre-defined event](http://scanimage.vidriotechnologies.com/display/SI2019/User+Functions). An example code to grab frames is available at the bottom of [this page](http://scanimage.vidriotechnologies.com/display/SI2019/ScanImage+API). Like in the case of Scanbox, we can write this data down into a memory-mapped file, which is being monitored by `improv`.

Example:
This example was run using MATLAB 2018b, ScanImage 5.5 with a [simulated DAQ](http://scanimage.vidriotechnologies.com/display/SI2019/How+to+Run+a+ScanImage+Simulation) using NI DAQmx 15.5. Tha machine file is `Machine_Data_File.m`. Copy everything in the `scanimage` folder into the current working folder. Run `initMemMap.m`, which creates 2 memory-mapped files.

<p align="center"><img width="500" alt="User Functions panel" src="https://user-images.githubusercontent.com/34997334/63535845-68b63200-c4e0-11e9-9e77-fd1970336761.png"></p>

Open the `User Functions` panel and load `usr_funcs.cfg`. ScanImage will call `initMemMap` upon the acquisition start and call `grabFrame` after every frame. Once we have a functioning system, we can use the MATLAB Engine API for Python to read the memmapfile as described above.

### Benchmark
<p align="center"><img width="600" alt="" src="https://user-images.githubusercontent.com/34997334/63537159-23473400-c4e3-11e9-8d13-7cdf1ab33c76.png"></p>
Time from `grabFrame` to NumPy array (512x512 int16). Core i7-4750HQ (Crystalwell, 3.1 GHz at the time of benchmark). Time scales linearly with data size.