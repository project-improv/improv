import os
import pathlib
from urllib import request

# TODO: replace directory strings with PATH
def download():
    """
    utility function that download source data from the web 
    Must be run from ./improv top level directory
    """
    datadir = pathlib.Path('demos/bubblewrap/data')
    filename = 'indy_20160407_02.mat'
    if not datadir.exists():
        print('data not found, downloading...')
        datadir.mkdir()
        os.chdir(str(datadir))
        request.urlretrieve('https://zenodo.org/record/3854034/files/indy_20160407_02.mat', filename='indy_20160407_02.mat')
        print('data downloaded')
    return 'data/'+filename # location of file

if __name__ == "__main__":
    download()