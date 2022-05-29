"""
Script to download dog model and sample video from DLC-live to run the demo.

Adapted from:

DeepLabCut Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
"""


import os
import shutil
from dlclive import benchmark_videos
import urllib.request


def main():

    # make temporary directory in $HOME
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # download dog test video from github:
    os.chdir("./data")
    url_link = "https://github.com/DeepLabCut/DeepLabCut-live/blob/master/check_install/dog_clip.avi?raw=True"
    urllib.request.urlretrieve(url_link, "dog_clip.avi")
    video_file = os.path.join(url_link, "dog_clip.avi")

    # download exported dog model from DeepLabCut Model Zoo
    os.chdir("../models")
    print("Downloading full_dog model from the DeepLabCut Model Zoo...")
    model_url = "http://deeplabcut.rowland.harvard.edu/models/DLC_Dog_resnet_50_iteration-0_shuffle-0.tar.gz"
    os.system(f"curl {model_url} | tar xvz")

if __name__ == "__main__":
    main()
