# Overview

Unit and integration testing is done using the **pytest** framework. Since parts of improv require asynchronous processing, the **pytest-asyncio** package is used in order to support tests that deal with asynchronous elements of the codebase. To measure testing coverage, we use the **pytest-cov** package. As of 10/27/2022, the versions of **pytest**, **pytest-asyncio**, and **pytest-cov** being used are versions 7.1.2, 0.19.0, and 3.0.0 respectively, running on python 3.9.0. Currently, there are 120 tests, 105 of which are fully written. For further reference regarding **pytest**, please visit their [website](https://docs.pytest.org/en/7.2.x/).

# Installation

To install **pytest**, run the following in the command line while in the environment you wish to use for testing: `pip install -U pytest`. Similarly, to install **pytest-asyncio** and **pytest-cov**, run `pip install pytest-asyncio` and `pip install pytest-cov`, respectively.

# Usage

Once **pytest** and all its related packages are installed, open the command line and navigate to the improv/pytest directory. Then, type `pytest` and enter to begin running the tests. You should see an output that looks like this:

![default_pytest_out](https://user-images.githubusercontent.com/104780909/198359858-9cfc3096-c771-46e4-a1f2-4e54ae640e3a.png)

To run the tests without warnings, use `pytest --disable-warnings`.

To run the tests with more information, use `pytest -vv`. The output should look something like this:

![verbose_pytest_out](https://user-images.githubusercontent.com/104780909/198360515-4a720ff4-1160-4f4f-8cef-130c9615cef0.png)

To run a specific test, use `pytest -k <test_name>`.

# Coverage

To examine test coverage, type the following command: `pytest --cov`. You should see something like this:

![cov_report](https://user-images.githubusercontent.com/104780909/198361418-5aede6fa-c406-4eea-b174-e0214c12de44.png)

# Timing

To get the duration of the first *n* tests, run the following: `pytest --duration=n`; e.g to get how long it takes for the 3 slowest tests to run, type the command `pytest --duration=3`.

# Currently Written Tests

Tests are currently written for the following python classes: Actor.py, Link.py, Nexus.py, Store.py, and Tweak.py. 

