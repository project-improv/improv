from demos.ava.scripts.models import VAE_Base, encoder, decoder

import torch

model_path = 'models/'

model = VAE_Base(encoder(32), decoder(32), model_path)

model.load_state('models/checkpoint_encoder_300.tar')
model.eval().to('cuda:0')

example_input = torch.rand((1,1,128,128)).to('cuda:0')
            
torch.jit.trace(model, example_input).save('models/vae.pt')