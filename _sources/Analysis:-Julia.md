Using [PyJulia](https://pyjulia.readthedocs.io/en/latest/), we could call functions and scripts in Julia directly from Python as shown in `analysis_julia.py`. 

### Calling Julia functions from Python
Initialization of the Julia backend may take up to 30 seconds. Pre-defined Julia functions could be loaded at startup by specifying the `julia_file` keyword in the configuration file. Upon loading, Julia functions could be wrapped and used as if it's a Python function in the following manner.

```python
j_func = julia.eval('julia_function')
j_func(2)
```
### Using NumPy array as a function input
When sending a NumPy array to Julia, PyJulia copies the entire array by default. This is safer but comes at a speed cost. By specifying the conversion type to `PyArray`, however, a zero-copy transfer can be achieved. For more details, check the Python object interfaces section in the PyCall [readme](https://github.com/JuliaPy/PyCall.jl/blob/master/README.md) (PyJulia is based on PyCall).

```python
j_func = julia.eval('pyfunction(julia_function, PyArray)')
j_func(np.zeros((100,100)
```
Array transfers from Julia, on the other hand, is zero-copy by default.

### Benchmark
<p align="center"><img width="600" alt="NumPy Transfer Benchmark" src="https://user-images.githubusercontent.com/34997334/63102946-1dcd7500-bf4a-11e9-8672-6b0f4948517e.png"></p>

Scaling of NumPy array transfer time from/to Julia. Error bar is 95% CI.