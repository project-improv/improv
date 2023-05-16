import os
import shutil
from dlclive import benchmark_videos
import urllib.request

custom_video = "data/raw_videos/hacker.mov"

# run inference and display keypoints on custom video
print("\n Running inference...\n")
model_dir = "data/models/DLC_Dog_resnet_50_iteration-0_shuffle-0"
print(custom_video)
benchmark_videos(
    model_dir,
    custom_video,
    save_video=True,
    output="data/analyzed_videos",
    display=False,
    resize=0.5,
    pcutoff=0.25,
)
