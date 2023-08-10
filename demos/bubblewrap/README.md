# Running Bubblewrap demo

This has only been tested Ubuntu. After installing improv, install additional dependencies (JAX on CPU by default) by being in the improv directory and do:

- `pip install -e ".[bw_demo]"`

To download the sample data from the paper, do:

- `python demos/bubblewrap/actors/utils.py`

This may take a few minutes. After data is downloaded, do:

- `python demos/bubblewrap/bubble_demo.py`


A GUI will pop up with two buttons named "setup" and "run". First hit "setup" and wait ~5 seconds, then hit "run". Bubblewrap will reduce the dimension of the neural data from ~180 neurons to 2 dimensions, represented by grey dots popping up on the blot, and coarsely tile the space with red bubbles. 

## For Windows Users

In order to have the GUI pop up, you must first install an X server. There are many possible options, but the one that has been tested is [VcXsrv](https://sourceforge.net/projects/vcxsrv/). For installation instructions, navigate back to the improv root directory and consult section 1, subsection III of the WSLINSTALL.md document. 
