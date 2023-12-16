# Running Bubblewrap demo
## Ubuntu users
After installing improv, install additional dependencies (JAX on CPU by default) by being in the improv directory and do:

- `pip install -e ".[bw_demo]"`

To download the sample data from the paper, do:

- `python demos/bubblewrap/actors/utils.py`

This may take a few minutes. After data is downloaded, run the GUI with:

- `python demos/bubblewrap/bubble_demo.py`

A GUI will pop up with two buttons named "setup" and "run". First hit "setup" and wait ~5 seconds, then hit "run". Bubblewrap will perform dimensionality reduction of ~180 neurons to 2 dimensions, represented by grey dots popping up on the plot, and coarsely tile the space with red bubbles to represent transitions in the low-dimension space. All in real-time! 

## Windows Users
Install dependencies and download data. Then, in order to have the GUI pop up, you must first install an X server. There are many possible options, but the one that has been tested is [VcXsrv](https://sourceforge.net/projects/vcxsrv/). For installation instructions, navigate back to the improv root directory and consult section 1, subsection III of the WSLINSTALL.md document. Then run GUI with `python demos/bubblewrap/bubble_demo.py`.
