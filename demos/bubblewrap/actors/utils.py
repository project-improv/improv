#%%  utility function that download source data from the web. Must be run from ./improv top level directory
import os
from urllib import request
def download():
    dir = 'demos/bubblewrap/data'
    if not os.path.exists(dir):
        os.makedirs(dir)
    os.chdir(dir)
    request.urlretrieve('https://zenodo.org/record/3854034/files/indy_20160407_02.mat', filename='indy_20160407_02.mat')