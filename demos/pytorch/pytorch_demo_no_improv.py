from demos.pytorch.actors.pytorch_processor import PyTorchProcessor
# Might be exact same as demos.neurofinder.actors.acquire_folder?
from demos.pytorch.actors.acquire_folder import FolderAcquirer

from improv.store import Limbo

import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import pickle
from PIL import Image

import subprocess

import sys
sys.path.append('/home/eao21/improv')

import torch
import time

import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Matplotlib is overly verbose by default
logging.getLogger("matplotlib").setLevel(logging.WARNING)

def plot_hist(data, title, out_plots_path, file_name, bins=5, range=None):
    plt.hist(data, bins=bins, range=range)
    plt.xlabel('Times (ms)', fontname='Arial')
    plt.ylabel('Frequency', fontname='Arial')
    plt.title(title)
    plt.show()

    plt.savefig(out_plots_path + file_name, dpi=300)

def main():
    ''' Demo PyTorch pipeline outside of improv for inputting CIFAR10 images into a pre-trained ResNet 50 model
    '''

    # 1. Set up vars:
    name = 'PyTorch'
    data_path = 'data/CIFAR10/'
    model_path = 'models/ResNet50-CIFAR10.pt'
    transforms_path = 'models/ResNet50-transforms.pt'
    config_file = 'pytorch_demo.yaml'

    out_timing_path = 'output/no_improv/timing'
    out_plots_path = 'output/no_improv/plots/'
    os.makedirs(out_timing_path exist_ok=True)
    os.makedirs(out_plots_path exist_ok=True)

    logger.info(data_path, model_path, out_timing_path, out_plots_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    pt_proc = PyTorchProcessor(name)
    model = pt_proc.setup(name, model_path, device, out_timing_path)
    transforms = pt_proc.setup(name, transforms_path, device, out_timing_path)

    img_acq = FolderAcquirer('Acquire Images from Folder', folder=data_path)

    path = Path(data_path)
    files = os.listdir(data_path)
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    total_imgs = len(files)

    load_img_time = []
    put_img_in = []
    get_img_out = []
    proc_img = []
    inference_time = []
    put_out_time = []
    per_img_time = []
    total_times = []

    t = time.time()
    for img_num in range(total_imgs):
        t1 = time.time()
        file = files[img_num]
        img = img_acq.get_img(data_path + file)
        time.sleep(lag)
        t2 = time.time()
        img_id = img_acq.put_img(client, img, img_num)
        t3 = time.time()
        img = limbo.getID(img_id)
        t4 = time.time()
        img = pt_proc.loadImage(img, transforms)
        t5 = time.time()
        output = pt_proc.runInference(img, model, device)
        t6 = time.time()
        if torch.is_tensor(output):
            out_id = client.put(pickle.dumps(output, protocol=pickle.HIGHEST_PROTOCOL), 'output-' + str(img_num))
        else:
            out_id = client.put(output, 'output-' + str(img_num))
        # ^^^ Same as below...
        # out_id = img_acq.put_img(output, 'output' + (img_num))
        # End put output into store
        t7 = time.time()

        load_img_time.append((t2 - t1)*1000.0)
        put_img_in.append((t3 - t2)*1000.0)
        get_img_out.append((t4 - t3)*1000.0)
        proc_img.append((t5 - t4)*1000.0)
        inference_time.append((t6 - t5)*1000.0)
        put_out_time.append((t7 - t6)*1000.0)
        per_img_time.append((t7 - t1)*1000.0)
    
    total_time = (time.time() - t)*1000.0
    print('Total time to run 200 CIFAR10 images: ', total_time)

    np.savetxt(out_timing_path + '/load_img_time.txt', np.array(load_img_time))
    np.savetxt(out_timing_path + '/put_img_in.txt', np.array(put_img_in))
    np.savetxt(out_timing_path +'/get_img_out.txt', np.array(get_img_out))
    np.savetxt(out_timing_path +'/proc_img.txt', np.array(proc_img))
    np.savetxt(out_timing_path +'/inference_time.txt', np.array(inference_time))
    np.savetxt(out_timing_path +'/put_out_time.txt', np.array(put_out_time))
    np.savetxt(out_timing_path +'/per_img_time.txt', np.array(per_img_time))

    plot_hist(load_img_time, 'Load Image from File Times', out_plots_path, 'load_img_times.png')
    plot_hist(put_img_in, 'Put Image in Store Times', out_plots_path, 'put_img_in.png')
    plot_hist(get_img_out, 'Get Image Out of Store Times', out_plots_path, 'get_img_out.png')
    plot_hist(proc_img, 'Image to Tensor Times', out_plots_path, 'proc_img.png')
    plot_hist(inference_time, 'Inference Times', out_plots_path, 'inference_time.png')
    plot_hist(put_out_time, 'Put Output in Store Times', out_plots_path, 'put_out_time.png')
    plot_hist(per_img_time, 'Total Times per Image', out_plots_path, 'per_img_time.png')

if __name__ == "__main__":
    main()

# %%
