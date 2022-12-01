# Prerequisites

`improv` depends on certain OS-specific libraries and these can be installed as followed.

### Ubuntu
   ```bash
   apt-get update
   apt-get install -y gcc g++ libgl1 libgl1-mesa-glx libqt5x11extras5
   ```

### macOS
   ```bash
   xcode-select --install
   ```

### Windows

Please consult [[installation for Windows]].

# Installation

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (a package manager for Python).

1. Install [mamba](https://mamba.readthedocs.io/en/latest/):
```
conda install -n base -c conda-forge mamba
```

1. Install `CaImAn`.
    ```bash
    git clone https://github.com/pearsonlab/CaImAn/
    cd CaImAn
    git checkout lite
    mamba env create -n improv -f environment.yml
    conda activate improv
    pip install .
    cd ..
    ```

1. Install `improv` (see also [[Building and Packaging]])
    ```bash
    mamba install pyqtgraph
    pip install build 
    git clone https://github.com/pearsonlab/improv
    cd improv
    python -m build 
    pip install -e .
    ````

1. Set the environmental variables for proper `CaImAn` functionality.
   ```bash
   export MKL_NUM_THREADS=1
   export OPENBLAS_NUM_THREADS=1
   ```

Note that the environmental variables has to be redefined every time a new shell is started. A more permanent solution would be to include these commands into the `.bashrc` or `.zshrc` file so that the variables are automatically redefined.
```bash
echo "export MKL_NUM_THREADS=1" >> ~/.bashrc
echo "export OPENBLAS_NUM_THREADS=1" >> ~/.bashrc
```
