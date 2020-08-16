import numpy as np
from sparl.datasets.hrtf import HRTF
from sparl.utils import ControlledNoise
from sparl.utils import PreEmphasis

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.datasets import LIBRISPEECH
from torchaudio.transforms import MelSpectrogram
from torchaudio.transforms import AmplitudeToDB
from torchaudio.transforms import MuLawEncoding


"""
Dataset class to manipulate the data from LIBRISPEECH
"""
class LibriSpeech(LIBRISPEECH):
    def __init__(
        self, ls_root, url, cipic_root, n_mels, 
        n_bits=8, seconds=.5, insert_noise=True):
        
        # reference to upper class
        super(LibriSpeech, self).__init__(root=ls_root, url=url)

        # Save passed parameters
        self.n_mels = n_mels
        self.pre_emphasis = PreEmphasis(alpha=.90)
        self.n_bits = n_bits
        self.insert_noise = insert_noise
        self.seconds = seconds

        # Constants
        self.hop_length = 160   # 10  ms
        self.n_fft = 2048       # 128 ms
        self.win_length = 400   # 25  ms
        self.fmin = 50
        self.sr = 16000
        self.top_db = 80
        self.samples = int(self.sr*self.seconds)

        # Audio operators
        self.hrtf = HRTF(cipic_root)
        self.subjects = [
            'subject_018', 'subject_009',
            'subject_003', 'subject_060',
            'subject_134'
        ]

        self.melspec = MelSpectrogram(
            sample_rate=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels= self.n_mels,
            win_length=self.win_length,
            f_min=self.fmin
        )

        self.amp_db = AmplitudeToDB(top_db=self.top_db)
        self.padding = nn.ConstantPad1d(96, 0.0)

        # Angles
        self.azimuthal = [
            -80, -65, -55, -45, -40, -35, -30, -25, -20,
            -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35,
            40, 45, 55, 65, 80
        ]
        self.elevations = [-45+5.625*x for x in range(50)]
        self.n_azimuthal = len(self.azimuthal)
        self.n_elevations = len(self.elevations)


    def __getitem__(self, idx):
        # Get item from the super class
        mono, _, _, _, _, _ = super(LibriSpeech, self).__getitem__(idx)
        offset = torch.randint(low=0, high=self.samples-1, size=(1,))
        mono = mono[:, offset:(offset+self.samples)]

        # Pre-emphasis filter
        mono = self.pre_emphasis(mono)
        mono = torch.clamp(mono, min=-1, max=1)

        # Insert random noise
        mono_noise = mono.clone()
        if self.insert_noise and (torch.rand(1) > .7):
            random_snr = torch.randint(low=15, high=30, size=(1,))
            noise_inj = ControlledNoise(random_snr)
            mono_noise = noise_inj(mono_noise)

        # Long representation
        mono_noise = mono_noise*(2**self.n_bits-1)
        mono_noise = mono_noise.type(torch.LongTensor)[0, :]

        az = np.random.randint(0, self.n_azimuthal)
        el = np.random.randint(0, self.n_elevations)
        y = self.get_direction(az, el)

        # Get subject HRIR
        subject = np.random.choice(self.subjects)
        audio = self.hrtf.convolve(mono_noise, self.sr, az, el, subject)
        audio = audio/(2**self.n_bits-1)
        
        # Convert to Torch Tensor
        audio = torch.Tensor(audio).type(torch.FloatTensor)
        audio = self.padding(audio)
        audio = torch.clamp(audio, min=-1, max=1)

        # Transform to melspectrogram
        left_spec = self.melspec(audio[0, :].unsqueeze(0))
        right_spec = self.melspec(audio[1, :].unsqueeze(0))

        # Log-Melspec normalized
        left_spec = self.amp_db(left_spec)/self.top_db + 1
        right_spec = self.amp_db(right_spec)/self.top_db + 1
        X = torch.cat([left_spec, right_spec], dim=0)

        parameters = {
            'angles': (az, el),
            'subject': subject
        }

        return X, torch.Tensor([y]), parameters

    def get_direction(self, az, el):
        return 0. if self.azimuthal[az] < 0 else 1.