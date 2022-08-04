import os
import torch
from resnet50 import ResNet50

gpu_num = 0

device = torch.device("cuda:{}".format(gpu_num) if torch.cuda.is_available() else "cpu")
                
model = ResNet50().to(device)
scripted_model = torch.jit.script(model).to(device)

os.makedirs("./models", exist_ok=True)
scripted_model.save("./models/ResNet50-GPU.pt")