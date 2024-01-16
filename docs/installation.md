(page:installation)=
# Installation and building

## Simple installation
The simplest way to install _improv_ is with pip:
```
pip install improv
```
````{warning}
Due to [this pyzmq issue](https://github.com/zeromq/libzmq/issues/3313), if you're running on Ubuntu, you need to specify
```
pip install improv --no-binary pyzmq
```
to build `pyzmq` from source.
````

## Optional dependencies
In addition to the basic _improv_ installation, users who want to, e.g., run tests locally and build docs should do
```
pip install improv[tests,docs]
```
(on bash) or
```
pip install "improv[tests,docs]"
```
(on zsh for newer Macs).

(page:installation:source_build)=
## Building from source
Users who want to run the demos or contribute to the project will want to build and install from source.

You can either obtain the source code from our [current releases](https://github.com/project-improv/improv/releases) (download and unzip the `Source code (zip)` file) or via cloning from GitHub:
```bash
git clone https://github.com/project-improv/improv.git
```

With the code downloaded, you'll then need to build and install the package. For installation, we recommend using a virtual environment so that _improv_ and its dependencies don't conflict with your existing Python setup. For instance, using [`mamba`](https://mamba.readthedocs.io/en/latest/):
```bash
mamba create -n improv python=3.10
```
will create an `improv` environment running Python 3.10 that you can activate via
```bash
mamba activate improv
```
Other options, such as [`virtualenv`](https://virtualenv.pypa.io/en/latest/) will also work. ([`conda`](https://docs.conda.io/projects/conda/en/stable/) is largely superseded by `mamba`.)

Currently, we build using [Setuptools](https://setuptools.pypa.io/en/latest/index.html) via a `pyproject.toml` file as specified in [PEP 518](https://peps.python.org/pep-0518/). This may allow us to switch out the build backend (or frontend) later.

For now, the package can be built by running
```
pip install --editable .
```
from within the project directory. 
this will install the package in editable mode, which means that changes in the project directory will affect the code that is run (i.e., the installation will not copy over the code to `site-packages` but simply link the project directory). 

When uninstalling, be sure to do so _from outside the project directory_, since otherwise, `pip` only appears to find the command line script, not the full package.


## Building documents locally
Currently, we have [GitHub Actions](https://docs.github.com/en/actions) set up to build the Jupyter Book [automatically](https://jupyterbook.org/en/stable/publish/gh-pages.html#automatically-host-your-book-with-github-actions). But for previewing changes locally, you will want to build the documents and check them yourself.

To do that, you'll first need to install the docs dependencies with
```
pip install improv[docs]
```
or
```
pip install -e .[docs]
```
Then simply run
```
jupyter-book build docs
```
and open `docs/_build/html/index.html` in your browser.

## Getting around certificate issues

On some systems, building (or installing from `pip`) can run into [this error](https://stackoverflow.com/questions/25981703/pip-install-fails-with-connection-error-ssl-certificate-verify-failed-certi) related to SSL certificates. For `pip`, the solution is given in the linked StackOverflow question (add `--trusted-host` to the command line), but for `build`, we run into the issue that the `trusted-host` flag will not be passed through to `pip`, and `pip` will build inside an isolated venv, meaning it won't read a file-based configuration option like the one given in the answer, either. 

The (inelegant) solution that will work is to set the `pip.conf` file with
```
[global]
trusted-host = pypi.python.org
               pypi.org
               files.pythonhosted.org
```
and then run
```
python -m build --no-isolation
```
which will allow `pip` to correctly read the configuration.