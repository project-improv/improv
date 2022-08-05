import time

import torch
import torch.multiprocessing as mp

import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

model_path = 'models/ResNet50.pt'
gpu_num = 0
device = torch.device("cuda:{}".format(gpu_num) if torch.cuda.is_available() else "cpu")

def setup(model_path, device):
    ''' Initialize model
    '''
    logger.info('Loading model from {}'.format(model_path))

    t = time.time()

    model = torch.jit.load(model_path).to(device)

    # model = torch.jit.load(model_path).to(device).share_memory

    load_model_time = time.time() - t
    print('Time to load model (multiprocessing): ', load_model_time*1000.0)
    with open("output/timing/mp_load_model_time.txt", "w") as text_file:
        text_file.write("%s" % load_model_time)

if __name__ == '__main__':
    # set_start_method() should not be used more than once in the program.
    # try:
    #     mp.set_start_method("forkserver")
    # except: pass
    q = mp.Queue
    p = mp.Process(target=setup, args=(model_path, device))
    p.start()
    # print(q.get())
    p.join()

# For consideration...maybe can use different start method (spawn/forkserver) for DL/CUDA init than native improv start method (fork?)
# Alternatively, you can use get_context() to obtain a context object. Context objects have the same API as the multiprocessing module, and allow one to use multiple start methods in the same program.

# NOTE:
# Note that objects related to one context may not be compatible with processes for a different context. In particular, locks created using the fork context cannot be passed to processes started using the spawn or forkserver start methods.

# A library which wants to use a particular start method should probably use get_context() to avoid interfering with the choice of the library user.

# Warning The 'spawn' and 'forkserver' start methods cannot currently be used with “frozen” executables (i.e., binaries produced by packages like PyInstaller and cx_Freeze) on Unix. The 'fork' start method does work.

# num_processes = 4
# processes = []
# for rank in range(num_processes):
#     p.start()
#     processes.append(p)
# for p in processes:
#     p.join()