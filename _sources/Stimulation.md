## Using ScanImage
Signals from `improv` can be sent to ScanImage using the MATLAB Engine API for Python. To enable access from Python, execute `matlab.engine.shareEngine` in the MATLAB instance where ScanImage is running.

All ScanImage GUIs have a variable correspondence in the code and can be explored using the ScanImage [API](http://scanimage.vidriotechnologies.com/display/SI2019/ScanImage+API). As mentioned in [Data Acquisition](Data Acquisition), MATLAB code could be executed directly as followed.

```python
eng = matlab.engine.connect_matlab(matlab.engine.find_matlab()[0])
eng.eval("hSI = evalin('base','hSI');", nargout=0)
eng.eval("hSI.hRoiManager.scanZoomFactor = 2", nargout=0)
```
In this example, the zoom factor is set to be 2.