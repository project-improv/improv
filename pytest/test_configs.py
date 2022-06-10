import pytest
import os
from improv.tweak import Tweak as twk

"""This test checks if all the config files are valid"""

def get_configs():
    os.chdir(os.getcwd() + "/./configs")
    cfgs = os.listdir()
    os.chdir(os.getcwd() + "/./..")
    return cfgs

@pytest.mark.skip
@pytest.mark.parametrize("configFile", get_configs())
def test_pkg_validity(configFile):
    t = twk(configFile)
    with not pytest.raises.ModuleNotFoundError:
        t.createConfig()
