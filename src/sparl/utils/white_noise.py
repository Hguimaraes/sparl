import torch
import torch.nn as nn
from torch.distributions import normal

"""
Helper class to insert white noise with desired SNR
"""
class ControlledNoise(nn.Module):
    def __init__(self, desired_snr_db=15):
        super(ControlledNoise, self).__init__()
        self.desired_snr_db = desired_snr_db
    
    def forward(self, x):
        x = x.view(-1)
        x = self.insert_controlled_noise(x, self.desired_snr_db)
        return x.view(1, -1)

    def insert_controlled_noise(self, signal, desired_snr_db=15):
        # Generate the noise
        n = signal.shape[0]
        sampler = normal.Normal(torch.tensor([.0]), torch.tensor([.1]))
        noise = sampler.sample(sample_shape=(1, n)).view(-1)

        # Calculate the power of signal and noise
        S_signal = signal.dot(signal) / n
        S_noise = noise.dot(noise) / n

        # Proportion factor
        K = (S_signal / S_noise)*(10**(-desired_snr_db/10.))

        # Rescale the noise
        new_noise = torch.sqrt(K)*noise

        return new_noise + signal