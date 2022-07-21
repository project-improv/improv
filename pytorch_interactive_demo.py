#%%

import sys # issue w/PYTHONPATH and sys.path...improv src not in there...modules not loading...

sys.path.append('/home/eao21/improv')

from pathlib import Path
import os
import subprocess

from improv.store import Limbo

import torch
from demos.pytorch.actors.pytorch_processor import PyTorchProcessor
# Might be exact same as demos.neurofinder.actors.acquire_folder?
from demos.pytorch.actors.acquire_folder import FolderAcquirer

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle

from numpy import linspace

import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Matplotlib is overly verbose by default
logging.getLogger("matplotlib").setLevel(logging.WARNING)

def plot_hist(data, title, out_plots_path, file_name, bins='auto', range=None):
    plt.hist(data, bins=bins, range=range)
    plt.xlabel('Times (ms)', fontname='Arial')
    plt.ylabel('Frequency', fontname='Arial')
    plt.title(title)
    plt.show()

    plt.savefig(out_plots_path + file_name, dpi=300)

''' Walk through pipeline outside of improv for pre-trained AleNet w/CIFAR10
    AlexNet and transforms = jit
    CIFAR10 = .jpg
    '''

# 1. Set up vars:
name = 'PyTorch Demo'
data_path = '/home/eao21/improv/demos/pytorch/data/CIFAR10/'
model_path = '/home/eao21/improv/demos/pytorch/models/ResNet50-CIFAR10.pt'
transforms_path = '/home/eao21/improv/demos/pytorch/models/ResNet50-transforms.pt'
# config_file = '/home/eao21/improv/demos/pytorch/pytorch_demo.yaml'

out_timing_path = '/home/eao21/improv/demos/pytorch/output/no_improv/timing'
# Might make separate script?
out_plots_path = '/home/eao21/improv/demos/pytorch/output/no_improv/plots/'
if not os._exists(out_timing_path):
    try:
        os.makedirs(out_timing_path)
    except:
        pass

if not os._exists(out_plots_path):
    try:
        os.makedirs(out_plots_path)
    except:
        pass

logger.info(data_path, model_path, out_timing_path, out_plots_path)

print('data_path: ', data_path, '\n',
'model_path: ', model_path, '\n',
'out_timing_path: ', out_timing_path, '\n',
'out_plots_path: ', out_plots_path, '\n')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 2. and 3. offline
# 2. Start store
subprocess.Popen(["plasma_store", "-s", "/tmp/store", "-m", str(10000000)],\
stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

limbo = Limbo()
client = limbo

# 3. Load model
# Default args for pt_proc = data_path, model_path, config_file, (all same as above), device=None
pt_proc = PyTorchProcessor(name)
model = pt_proc.setup(name, model_path, device, out_timing_path, msg='Load model time: ')
transforms = pt_proc.setup(name, transforms_path, device, out_timing_path, msg='Load transforms model time: ')
# Prints out load time

# 4. runProcess - loop through images
# A. Get image - load folder offline, don't time, time loop to get, process, put one-by-one
img_acq = FolderAcquirer('Acquire Images from Folder', folder=data_path)
lag = 0.005

path = Path(data_path)
files = os.listdir(data_path)
files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

# For the loop
total_imgs = len(files)

load_img_time = []
put_img_in = []
get_img_out = []
proc_img = []
inference_time = []
put_out_time = []
per_img_time = []
total_times = []

# Total time it takes to run through all images
t = time.time()
# For all images in folder (100 CIFAR10)
for img_num in range(total_imgs):
    # Start load image from file...not necessary to time, but so what...
    t1 = time.time()
    # Camel case?
    file = files[img_num]
    ## Lazy way to avoid pickling PIL image?
    img = img_acq.get_img(data_path + file)
    # End load image from file
    # Start put image into store
    time.sleep(lag)
    t2 = time.time()
    # Camel case?
    # TODO: update to pickle PIL image, or any way we would like to read image data...see .tiff
    img_id = img_acq.put_img(client, img, img_num)
    # End put image into store
    # Start get image from store
    t3 = time.time()
    img = limbo.getID(img_id)
    # End get image from store - should be PIL
    # Start image -> PIL -> tensor
    t4 = time.time()
    img = pt_proc.loadImage(img, transforms)
    # End image to tensor
    # Start inference, input -> tensor
    t5 = time.time()
    output = pt_proc.runInference(img, model, device)
    # End inference
    # Start put output into store
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
plot_hist(proc_img[1:], 'Image to Tensor Times', out_plots_path, 'proc_img.png')
plot_hist(inference_time[1:], 'Inference Times', out_plots_path, 'inference_time.png')
plot_hist(put_out_time, 'Put Output in Store Times', out_plots_path, 'put_out_time.png')
plot_hist(per_img_time[1:], 'Total Times per Image', out_plots_path, 'per_img_time.png')
# %%
