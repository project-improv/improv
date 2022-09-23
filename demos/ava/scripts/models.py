import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Normal, MultivariateNormal, LowRankMultivariateNormal
import os
import numpy as np
import matplotlib.pyplot as plt


class encoder(nn.Module):

	def __init__(self,z_dim=32):

		"""
		encoder for birdsong VAEs
		"""

		super(encoder,self).__init__()

		self.z_dim = z_dim

		self.encoder_conv = nn.Sequential(nn.BatchNorm2d(1),
								nn.Conv2d(1, 8, 3,1,padding=1),
								nn.ReLU(),
								nn.BatchNorm2d(8),
								nn.Conv2d(8, 8, 3,2,padding=1),
								nn.ReLU(),
								nn.BatchNorm2d(8),
								nn.Conv2d(8, 16,3,1,padding=1),
								nn.ReLU(),
								nn.BatchNorm2d(16),
								nn.Conv2d(16,16,3,2,padding=1),
								nn.ReLU(),
								nn.BatchNorm2d(16),
								nn.Conv2d(16,24,3,1,padding=1),
								nn.ReLU(),
								nn.BatchNorm2d(24),
								nn.Conv2d(24,24,3,2,padding=1),
								nn.ReLU(),
								nn.BatchNorm2d(24),
								nn.Conv2d(24,32,3,1,padding=1),
								nn.ReLU())

		self.encoder_fc = nn.Sequential(nn.Linear(8193,1024),
								nn.ReLU(),
								nn.Linear(1024,256),
								nn.ReLU())

		self.fc11 = nn.Linear(256,64)
		self.fc12 = nn.Linear(256,64)
		self.fc13 = nn.Linear(256,64)
		self.fc21 = nn.Linear(64,self.z_dim)
		self.fc22 = nn.Linear(64,self.z_dim)
		self.fc23 = nn.Linear(64,self.z_dim)
		
	def encode(self,x):

		h = self.encoder_conv(x)
		h = h.view(-1,8192)
		#print(h.shape)
		#print(torch.zeros(h.shape[0],1,device=h.device).shape)
		dummy = torch.zeros(h.shape[0],1,device=h.device)
		h = torch.cat((h,dummy),dim=1)
		#print(h.shape)
		h = self.encoder_fc(h)
		mu = F.relu(self.fc11(h))
		u = F.relu(self.fc12(h))
		d = F.relu(self.fc13(h))
		mu = self.fc21(mu)
		u = self.fc22(u)
		d = self.fc23(d)
		
		return mu, u.unsqueeze(-1),d.exp()

	def encode_with_time(self,x,encode_times):

		h = self.encoder_conv(x)
		h = h.view(-1,8192)
		
		h = torch.cat((h,encode_times.unsqueeze(1)),dim=1)
		h = self.encoder_fc(h)
		mu = F.relu(self.fc11(h))
		u = F.relu(self.fc12(h))
		d = F.relu(self.fc13(h))
		mu = self.fc21(mu)
		u = self.fc22(u)
		d = self.fc23(d)
		
		return mu, u.unsqueeze(-1),d.exp()

	def sample_z(self,mu,u,d):

		dist = LowRankMultivariateNormal(mu,u,d)

		z_hat = dist.rsample()

		return z_hat


class decoder(nn.Module):

	def __init__(self, z_dim=32,precision =1e4):
		"""
		Initialize stupid decoder

		Inputs
		-----
			z_dim: int, dim of latent dimension
			x_dim: int, dim of input data
			decoder_dist: bool, determines if we learn var of decoder in addition
						to mean
		"""

		super(decoder,self).__init__()
		self.precision = precision
		self.decoder_fc = nn.Sequential(nn.Linear(z_dim,64),
										nn.Linear(64,256),
										nn.Linear(256,1024),
										nn.ReLU(),
										nn.Linear(1024,8193),
										nn.ReLU())
		self.decoder_convt = nn.Sequential(nn.BatchNorm2d(32),
										nn.ConvTranspose2d(32,24,3,1,padding=1),
										nn.ReLU(),
										nn.BatchNorm2d(24),
										nn.ConvTranspose2d(24,24,3,2,padding=1,output_padding=1),
										nn.ReLU(),
										nn.BatchNorm2d(24),
										nn.ConvTranspose2d(24,16,3,1,padding=1),
										nn.ReLU(),
										nn.BatchNorm2d(16),
										nn.ConvTranspose2d(16,16,3,2,padding=1,output_padding=1),
										nn.ReLU(),
										nn.BatchNorm2d(16),
										nn.ConvTranspose2d(16,8,3,1,padding=1),
										nn.ReLU(),
										nn.BatchNorm2d(8),
										nn.ConvTranspose2d(8,8,3,2,padding=1,output_padding=1),
										nn.ReLU(),
										nn.BatchNorm2d(8),
										nn.ConvTranspose2d(8,1,3,1,padding=1))


		#device_name = "cuda" if torch.cuda.is_available() else "cpu"
		#self.device = torch.device(device_name)

		#self.to(self.device)


	def decode(self,z,return_time = False):
		"""
		Decode latent samples

		Inputs
		-----
			z: torch.tensor, latent samples to be decoded

		Outputs
		-----
			mu: torch.tensor, mean of decoded distribution
			if decoder_dist:
				logvar: torch.tensor, logvar of decoded distribution
		"""
		#print(z.dtype)
		z = self.decoder_fc(z)
		that = z[:,-1]
		z = z[:,:-1]
		z = z.view(-1,32,16,16)
		xhat = self.decoder_convt(z)
		#mu = self.mu_convt(z)

		if return_time:
			return xhat, that
		else:
			return xhat

class VAE_Base(nn.Module):

	def __init__(self, encoder, decoder,save_dir,lr=1e-4):

		super(VAE_Base,self).__init__()

		self.encoder = encoder 
		self.decoder = decoder 
		self.z_dim = self.encoder.z_dim
		self.save_dir = save_dir 
		self.epoch = 0
		self.lr = lr
		self.loss = {'train': {}, 'test': {}}

		device_name = "cuda" if torch.cuda.is_available() else "cpu"
		self.device = torch.device(device_name)
		self.to(self.device)
		self.optimizer = Adam(self.parameters(), lr=lr) # 8e-5


	def _compute_reconstruction_loss(self,x,xhat):

		constant = -0.5 * np.prod(x.shape[1:]) * np.log(2*np.pi/self.decoder.precision)

		x = torch.reshape(x,(x.shape[0],-1))
		xhat = torch.reshape(xhat,(xhat.shape[0],-1))
		l2s = torch.sum(torch.pow(x - xhat,2),axis=1)

		logprob = constant - 0.5 * self.decoder.precision * torch.sum(l2s)

		return logprob

	def _compute_kl_loss(self,mu,u,d):

		## uses matrix determinant lemma to compute logdet of covar
		Ainv = torch.diag_embed(1/d)
		#print(Ainv.shape)
		#print(u.shape)
		term1 = torch.log(1 + u.transpose(-2,-1) @ Ainv @ u)
		term2 = torch.log(d).sum()

		ld = term1 + term2 
		ld = ld.squeeze()
		mean = torch.pow(mu,2)

		#print((u @ u.transpose(-2,-1)).shape)
		#print((torch.diag_embed(d)).shape)
		trace = torch.diagonal((u @ u.transpose(-2,-1)) \
					+ torch.diag_embed(d),dim1=-2,dim2=-1)

		kl = 0.5 * ((trace + mean - 1).sum(axis=1) - ld).sum()
		#print(trace.shape)
		#print(mean.shape)
		#print(ld.shape)
		#print(kl.shape)
		return kl


	def compute_loss(self,x,return_recon = False):


		mu,u,d = self.encoder.encode(x)

		dist = LowRankMultivariateNormal(mu,u,d)

		zhat = dist.rsample()

		xhat = self.decoder.decode(zhat)

		kl = self._compute_kl_loss(mu,u,d)
		logprob = self._compute_reconstruction_loss(x,xhat)
		#print(kl.shape)
		#print(logprob.shape)
		elbo = logprob - kl 
		
		if return_recon:
			return -elbo,logprob, kl, xhat.view(-1,128,128).detach().cpu().numpy() 
		else:
			return -elbo,logprob, kl


	def train_epoch(self,train_loader):

		self.train()

		train_loss = 0.0
		train_kl = 0.0
		train_lp = 0.0 

		for ind, batch in enumerate(train_loader):
			#print(batch)
			self.optimizer.zero_grad()
			(spec,day) = batch 
			day = day.to(self.device).squeeze()
			#print(day.shape)
			#print(spec.shape)
			#spec = torch.stack(spec,axis=0)
			spec = spec.to(self.device).squeeze().unsqueeze(1)

			loss,lp,kl = self.compute_loss(spec)

			train_loss += loss.item()
			train_kl += kl.item()
			train_lp += lp.item()

			loss.backward()
			self.optimizer.step()

		train_loss /= len(train_loader)
		train_kl /= len(train_loader)
		train_lp /= len(train_loader)

		print('Epoch {0:d} average train loss: {1:.3f}'.format(self.epoch,train_loss))
		print('Epoch {0:d} average train kl: {1:.3f}'.format(self.epoch,train_kl))
		print('Epoch {0:d} average train lp: {1:.3f}'.format(self.epoch,train_lp))

		return train_loss

	def test_epoch(self,test_loader):

		self.eval()
		test_loss = 0.0
		test_kl = 0.0
		test_lp = 0.0 

		for ind, batch in enumerate(test_loader):

			(spec,day) = batch 
			day = day.to(self.device).squeeze()

			#spec = torch.stack(spec,axis=0)
			spec = spec.to(self.device).squeeze().unsqueeze(1)
			with torch.no_grad():
				loss,lp,kl = self.compute_loss(spec)

			test_loss += loss.item()
			test_kl += kl.item()
			test_lp += lp.item()

			
		test_loss /= len(test_loader)
		test_kl /= len(test_loader)
		test_lp /= len(test_loader)

		print('Epoch {0:d} average test loss: {1:.3f}'.format(self.epoch,test_loss))
		print('Epoch {0:d} average test kl: {1:.3f}'.format(self.epoch,test_kl))
		print('Epoch {0:d} average test lp: {1:.3f}'.format(self.epoch,test_lp))

		return test_loss

	def visualize(self, loader, num_specs=5):
		"""
		Plot spectrograms and their reconstructions.

		Spectrograms are chosen at random from the Dataloader Dataset.

		Parameters
		----------
		loader : torch.utils.data.Dataloader
			Spectrogram Dataloader
		num_specs : int, optional
			Number of spectrogram pairs to plot. Defaults to ``5``.
		gap : int or tuple of two ints, optional
			The vertical and horizontal gap between images, in pixels. Defaults
			to ``(2,6)``.
		save_filename : str, optional
			Where to save the plot, relative to `self.save_dir`. Defaults to
			``'temp.pdf'``.

		Returns
		-------
		specs : numpy.ndarray
			Spectgorams from `loader`.
		rec_specs : numpy.ndarray
			Corresponding spectrogram reconstructions.
		"""
		# Collect random indices.
		assert num_specs <= len(loader.dataset) and num_specs >= 1
		indices = np.random.choice(np.arange(len(loader.dataset)),
			size=num_specs,replace=False)
		
		(specs,days) = loader.dataset[indices]
		for spec in specs:
			spec = spec.to(self.device).squeeze().unsqueeze(1)


			# Retrieve spectrograms from the loader.
			# Get resonstructions.
			with torch.no_grad():
				_, _, _,rec_specs = self.compute_loss(spec, return_recon=True)
			spec = spec.detach().cpu().numpy()
			nrows = 1 + spec.shape[0]//5
			fig,axs = plt.subplots(nrows=nrows,ncols=5)
			row_ind = 0
			col_ind = 0

			for im in range(rec_specs.shape[0]):

				if col_ind >= 5:
					row_ind += 1
					col_ind = 0

				axs[row_ind,col_ind].imshow(rec_specs[im,:,:],origin='lower')
				axs[row_ind,col_ind].get_xaxis().set_visible(False)
				axs[row_ind,col_ind].get_yaxis().set_visible(False)
				col_ind += 1

			for ii in range(col_ind,5):
				axs[row_ind,ii].get_xaxis().set_visible(False)
				axs[row_ind,ii].get_yaxis().set_visible(False)
				axs[row_ind,ii].axis('square')


			#all_specs = np.stack([specs, rec_specs])
		# Plot.
			save_fn = 'reconstruction_epoch_' + str(self.epoch) + '_' + str(im) + '.png' 
			save_filename = os.path.join(self.save_dir, save_fn)

			plt.savefig(save_filename)
			plt.close('all')

			fig,axs = plt.subplots(nrows=nrows,ncols=5)
			row_ind = 0
			col_ind = 0

			for im in range(spec.shape[0]):

				if col_ind >= 5:
					row_ind += 1
					col_ind = 0

				axs[row_ind,col_ind].imshow(spec[im,:,:,:].squeeze(),origin='lower')
				axs[row_ind,col_ind].get_xaxis().set_visible(False)
				axs[row_ind,col_ind].get_yaxis().set_visible(False)
				col_ind += 1

			for ii in range(col_ind,5):
				axs[row_ind,ii].get_xaxis().set_visible(False)
				axs[row_ind,ii].get_yaxis().set_visible(False)
				axs[row_ind,ii].axis('square')

			#all_specs = np.stack([specs, rec_specs])
		# Plot.
			save_fn = 'real_epoch_' + str(self.epoch) + '_' + str(im) + '.png' 
			save_filename = os.path.join(self.save_dir, save_fn)

			plt.savefig(save_filename)
			plt.close('all')

		return 

	def train_test_loop(self,loaders, epochs=100, test_freq=2, save_freq=10,
		vis_freq=1):
		"""
		Train the model for multiple epochs, testing and saving along the way.

		Parameters
		----------
		loaders : dictionary
			Dictionary mapping the keys ``'test'`` and ``'train'`` to respective
			torch.utils.data.Dataloader objects.
		epochs : int, optional
			Number of (possibly additional) epochs to train the model for.
			Defaults to ``100``.
		test_freq : int, optional
			Testing is performed every `test_freq` epochs. Defaults to ``2``.
		save_freq : int, optional
			The model is saved every `save_freq` epochs. Defaults to ``10``.
		vis_freq : int, optional
			Syllable reconstructions are plotted every `vis_freq` epochs.
			Defaults to ``1``.
		"""
		print("="*40)
		print("Training: epochs", self.epoch, "to", self.epoch+epochs-1)
		print("Training set:", len(loaders['train'].dataset))
		print("Test set:", len(loaders['test'].dataset))
		print("="*40)
		# For some number of epochs...
		for epoch in range(self.epoch, self.epoch+epochs):
			# Run through the training data and record a loss.
			loss = self.train_epoch(loaders['train'])
			self.loss['train'][epoch] = loss
			# Run through the test data and record a loss.
			if (test_freq is not None) and (epoch % test_freq == 0):
				loss = self.test_epoch(loaders['test'])
				self.loss['test'][epoch] = loss
			# Save the model.
			if (save_freq is not None) and (epoch % save_freq == 0) and \
					(epoch > 0):
				filename = "checkpoint_"+str(epoch).zfill(3)+'.tar'
				self.save_state()
			# Plot reconstructions.
			if (vis_freq is not None) and (epoch % vis_freq == 0):
				self.visualize(loaders['test'])

			self.epoch += 1

	def get_latent(self,loader):

		latents = []

		for ind, batch in enumerate(loader):

			(spec,day) = batch 
			day = day.to(self.device)

			spec = spec.to(self.device).squeeze().unsqueeze(1)
			with torch.no_grad():
				z_mu,_,_ = self.encoder.encode(spec)

			latents.append(z_mu.detach().cpu().numpy())

		#latents = np.vstack(latents)
		return latents
		
	def forward(self, spec):

		# latents = []
		
		# spec = spec.to(self.device).squeeze().unsqueeze(1)
		
		with torch.no_grad():
			z_mu,_,_ = self.encoder.encode(spec)

			# latents.append(z_mu.detach().cpu().numpy())

		#latents = np.vstack(latents)
		return z_mu
		
	def save_state(self):

		"""
		Save state of network. Saves encoder and decoder state separately. Assumes
		that while training, you have been using set_epoch to set the epoch
		"""

		#self.set_epoch(epoch)
		fn = os.path.join(self.save_dir, 'checkpoint_encoder_' + str(self.epoch) + '.tar')

		"""Save state."""
		sd = self.state_dict()
		torch.save({
				'model_state_dict': sd,
				'optimizer_state_dict': self.optimizer.state_dict(),
				'epoch': self.epoch
			}, fn)

	def load_state(self,fn):

		"""
		Load state of network. Requires an epoch to recover the current state of network

		Inputs:
		-----
			epoch: int, current epoch of training
		"""
		"""Load state."""

		print("Loading state from:", fn)
		#print(self.state_dict().keys())

		checkpoint = torch.load(fn, map_location=self.device)
		#layer_1 = checkpoint['model_state_dict'].pop('layer_1')

		self.load_state_dict(checkpoint['model_state_dict'])
		self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		self.epoch = checkpoint['epoch']

class SmoothnessPriorVae(VAE_Base):

	def __init__(self, encoder, decoder,save_dir,lr=1e-4):

		super(SmoothnessPriorVae,self).__init__(encoder, decoder,save_dir,lr=1e-4)

	def _compute_kl_loss(self, mu, u, d):

		Ainv = torch.diag_embed(1/d)
		#print(Ainv.shape)
		#print(u.shape)
		term1 = torch.log(1 + u.transpose(-2,-1) @ Ainv @ u)
		term2 = torch.log(d).sum()

		ld = term1 + term2 
		ld = ld.squeeze()

		mu = torch.cat([torch.zeros(1,mu.shape[1],device=self.device),mu],axis=0)
		mean = torch.pow(mu[:-1,:] - mu[1:,:],2)

		#print((u @ u.transpose(-2,-1)).shape)
		#print((torch.diag_embed(d)).shape)
		trace = torch.diagonal((u @ u.transpose(-2,-1)) \
					+ torch.diag_embed(d),dim1=-2,dim2=-1)

		kl = 0.5 * ((trace + mean - 1).sum(axis=1) - ld).sum()

		return kl

class ReconstructTimeVae(VAE_Base):

	def __init__(self, encoder, decoder,save_dir,lr=1e-4):

		super(ReconstructTimeVae,self).__init__(encoder, decoder,save_dir,lr=1e-4)

	def compute_loss(self,x,encode_times,return_recon = False,weight=128**3):


		mu,u,d = self.encoder.encode_with_time(x,encode_times)

		dist = LowRankMultivariateNormal(mu,u,d)

		zhat = dist.rsample()

		xhat,that = self.decoder.decode(zhat,return_time=True)

		kl = self._compute_kl_loss(mu,u,d).sum()
		logprob = self._compute_reconstruction_loss(x,xhat).sum()

		time_reg = weight * torch.pow(that - encode_times,2).sum()

		elbo = logprob - kl 

		if return_recon:
			return -elbo + time_reg,logprob, kl, time_reg, xhat.view(-1,128,128).detach().cpu().numpy() 
		else:
			return -elbo + time_reg,logprob, kl, time_reg


	def train_epoch(self,train_loader):

		self.train()

		train_loss = 0.0
		train_kl = 0.0
		train_lp = 0.0
		train_tr = 0.0

		dt = train_loader.dataset.dt 

		for ind, batch in enumerate(train_loader):

			self.optimizer.zero_grad()
			(spec,day) = batch 
			day = day.to(self.device).squeeze()

			#spec = torch.stack(spec,axis=0)
			spec = spec.to(self.device).squeeze().unsqueeze(1)
			start_time = 0.0
			end_time = dt * spec.shape[0]
			encode_times = torch.arange(start_time,end_time,dt,device=self.device)
			loss,lp,kl,tr = self.compute_loss(spec,encode_times)

			train_loss += loss.item()
			train_kl += kl.item()
			train_lp += lp.item()
			train_tr += tr.item()

			loss.backward()
			self.optimizer.step()

		train_loss /= len(train_loader)
		train_kl /= len(train_loader)
		train_lp /= len(train_loader)
		train_tr /= len(train_loader)

		print('Epoch {0:d} average train loss: {1:.3f}'.format(self.epoch,train_loss))
		print('Epoch {0:d} average train kl: {1:.3f}'.format(self.epoch,train_kl))
		print('Epoch {0:d} average train lp: {1:.3f}'.format(self.epoch,train_lp))
		print('Epoch {0:d} average train time rec: {1:.3f}'.format(self.epoch,train_tr))

		return train_loss

	def test_epoch(self,test_loader):

		self.eval()
		test_loss = 0.0
		test_kl = 0.0
		test_lp = 0.0 
		test_tr = 0.0
		dt = test_loader.dataset.dt

		for ind, batch in enumerate(test_loader):

			(spec,day) = batch 
			day = day.to(self.device).squeeze()

			#spec = torch.stack(spec,axis=0)
			spec = spec.to(self.device).squeeze().unsqueeze(1)
			start_time = 0.0
			end_time = dt * spec.shape[0]
			encode_times = torch.arange(start_time,end_time,dt,device=self.device)
			with torch.no_grad():
				loss,lp,kl,tr = self.compute_loss(spec,encode_times)

			test_loss += loss.item()
			test_kl += kl.item()
			test_lp += lp.item()
			test_tr += tr.item()

			
		test_loss /= len(test_loader)
		test_kl /= len(test_loader)
		test_lp /= len(test_loader)
		test_tr /= len(test_loader)

		print('Epoch {0:d} average test loss: {1:.3f}'.format(self.epoch,test_loss))
		print('Epoch {0:d} average test kl: {1:.3f}'.format(self.epoch,test_kl))
		print('Epoch {0:d} average test lp: {1:.3f}'.format(self.epoch,test_lp))
		print('Epoch {0:d} average test tr: {1:.3f}'.format(self.epoch,test_tr))

		return test_loss

	def visualize(self, loader, num_specs=5):
		"""
		Plot spectrograms and their reconstructions.

		Spectrograms are chosen at random from the Dataloader Dataset.

		Parameters
		----------
		loader : torch.utils.data.Dataloader
			Spectrogram Dataloader
		num_specs : int, optional
			Number of spectrogram pairs to plot. Defaults to ``5``.
		gap : int or tuple of two ints, optional
			The vertical and horizontal gap between images, in pixels. Defaults
			to ``(2,6)``.
		save_filename : str, optional
			Where to save the plot, relative to `self.save_dir`. Defaults to
			``'temp.pdf'``.

		Returns
		-------
		specs : numpy.ndarray
			Spectgorams from `loader`.
		rec_specs : numpy.ndarray
			Corresponding spectrogram reconstructions.
		"""
		# Collect random indices.
		dt = loader.dataset.dt
		assert num_specs <= len(loader.dataset) and num_specs >= 1
		indices = np.random.choice(np.arange(len(loader.dataset)),
			size=num_specs,replace=False)
		
		(specs,days) = loader.dataset[indices]
		for spec in specs:
			spec = spec.to(self.device).squeeze().unsqueeze(1)


			# Retrieve spectrograms from the loader.
			# Get resonstructions.
			start_time = 0.0
			end_time = dt * spec.shape[0]
			encode_times = torch.arange(start_time,end_time,dt,device=self.device)
			with torch.no_grad():
				_, _, _,_,rec_specs = self.compute_loss(spec, encode_times,return_recon=True)
			spec = spec.detach().cpu().numpy()
			nrows = 1 + spec.shape[0]//5
			fig,axs = plt.subplots(nrows=nrows,ncols=5)
			row_ind = 0
			col_ind = 0

			for im in range(rec_specs.shape[0]):

				if col_ind >= 5:
					row_ind +=1
					col_ind = 0
				axs[row_ind,col_ind].imshow(rec_specs[im,:,:],origin='lower')
				axs[row_ind,col_ind].get_xaxis().set_visible(False)
				axs[row_ind,col_ind].get_yaxis().set_visible(False)
				col_ind += 1

			for ii in range(col_ind,5):
				axs[row_ind,ii].get_xaxis().set_visible(False)
				axs[row_ind,ii].get_yaxis().set_visible(False)
				axs[row_ind,ii].axis('square')

			#all_specs = np.stack([specs, rec_specs])
		# Plot.
			save_fn = 'reconstruction_epoch_' + str(self.epoch) + '_' + str(im) + '.png' 
			save_filename = os.path.join(self.save_dir, save_fn)

			plt.savefig(save_filename)
			plt.close('all')

			fig,axs = plt.subplots(nrows=nrows,ncols=5)
			row_ind = 0
			col_ind = 0

			for im in range(spec.shape[0]):

				if col_ind >= 5:
					row_ind +=1
					col_ind = 0
				axs[row_ind,col_ind].imshow(spec[im,:,:,:].squeeze(),origin='lower')
				axs[row_ind,col_ind].get_xaxis().set_visible(False)
				axs[row_ind,col_ind].get_yaxis().set_visible(False)
				col_ind += 1

			for ii in range(col_ind,5):
				axs[row_ind,ii].get_xaxis().set_visible(False)
				axs[row_ind,ii].get_yaxis().set_visible(False)
				axs[row_ind,ii].axis('square')

			#all_specs = np.stack([specs, rec_specs])
		# Plot.
			save_fn = 'real_epoch_' + str(self.epoch) + '_' + str(im) + '.png' 
			save_filename = os.path.join(self.save_dir, save_fn)

			plt.savefig(save_filename)
			plt.close('all')

		return 

	def get_latent(self,loader):

		latents = []
		dt = loader.dataset.dt

		for ind, batch in enumerate(loader):

			(spec,day) = batch 
			day = day.to(self.device)

			spec = spec.to(self.device).squeeze().unsqueeze(1)
			start_time = 0.0
			end_time = dt * spec.shape[0]
			encode_times = torch.arange(start_time,end_time,dt,device=self.device)

			with torch.no_grad():
				z_mu,_,_ = self.encoder.encode_with_time(spec,encode_times)

			latents.append(z_mu.detach().cpu().numpy())

		#latents = np.vstack(latents)
		return latents

if __name__ == '__main__':

	pass