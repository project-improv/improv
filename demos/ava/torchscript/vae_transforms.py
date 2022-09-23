import torch
from torch import nn
import torchaudio.transforms as TA
import torchvision.transforms as TV

from demos.ava.scripts.models import VAE_Base, encoder, decoder

import time

class VAE(nn.Module):

    def __init__(self, device, z_dim, checkpoint, params):
        super().__init__()
        self.device = device
        self.model = VAE_Base(encoder(z_dim), decoder(z_dim), '').eval().to(self.device)
        self.model.load_state(checkpoint)

        # transforms = torch.nn.Sequential(
        #     TA.Spectrogram(
        #         n_fft=n_fft,
        #         win_length=params['nperseg'],
        #         hop_length=params['noverlap'],
        #         center=True,
        #         power=2.0),
        #     TA.MelScale(
        #         sample_rate=fs,
        #         f_min = params['f_min'], f_max = params['f_max'], 
        #         n_fft=n_fft)).to(self.device)

        self.spec = TA.Spectrogram(n_fft=params['n_fft'],win_length=params['win_len'],
        hop_length=params['noverlap'],
        center=True,
        power=2.0).to(self.device)

        self.spec_img = TV.Resize((params['height'], params['width'])).to(self.device)

    def forward(self, x):
        '''
        '''
        with torch.no_grad():
            t = time.time()
            x = x.to(self.device)
            torch.cuda.synchronize()
            t_dev = [time.time() - t]
            spec = self.spec(x).unsqueeze(0)
            torch.cuda.synchronize()
            t_spec = torch.Tensor(time.time() - t_dev[0])
            spec_img = self.spec_img(spec).unsqueeze(0)
            torch.cuda.synchronize()
            t_spec_img = [time.time() - t_spec[0]]
            latents = self.model(spec_img)
            torch.cuda.synchronize()
            t_inf = [time.time() - t_spec_img[0]]
            
            return spec, spec_img, latents, t_dev, t_spec, t_spec_img, t_inf

# Add reconstruction -> same pass?

model_path = 'models/'
checkpoint = 'models/checkpoint_encoder_300.tar'

device = 'cuda:0'

z_dim = 32

params = {'width': 128,
        'height': 128, 
        'min_freq': 400,
        'max_freq': 16000,
        'win_len': 1,
        'n_fft': 512,
        'noverlap': 256,
        'spec_min_val': 2.0 ,
        'spec_max_val': 6.0}

model = VAE(device, z_dim=32, checkpoint=checkpoint, params=params)

example_input = torch.rand((640)).to(device)
            
torch.jit.trace(model, example_input).save('models/vae_transforms.pt')