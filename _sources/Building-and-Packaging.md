# With pip:

Currently, we are building using [Setuptools](https://setuptools.pypa.io/en/latest/index.html) via a `pyproject.toml` file as specified in [PEP 518](https://peps.python.org/pep-0518/). This may allow us to switch out the build backend (or frontend) later.

For now, the package can be built by running
```
python -m build
```
from within the project directory. After that
```
pip install --editable .
```
will install the package in editable mode, which means that changes in the project directory will affect the code that is run (i.e., the installation will not copy over the code to `site-packages` but simply link the project directory. 

When uninstalling, be sure to do so _from outside the project directory_, since otherwise, `pip` only appears to find the command line script, not the full package.