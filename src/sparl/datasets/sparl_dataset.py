import numpy as np
from sparl.datasets.hrtf import HRTF
from sparl.datasets.waveform import sine_wave
from sparl.datasets.waveform import square_wave
from sparl.datasets.waveform import triangle_wave
from sparl.utils import ControlledNoise

import torch
import torch.nn as nn
from torch.utils import data
from torchaudio.transforms import MelSpectrogram
from torchaudio.transforms import AmplitudeToDB


class SPARL(data.Dataset):
    def __init__(self, root, n_mels=128, min_freq=100, 
        max_freq=3100, sr=16000, seconds=.5, n_bits=8, is_validation=False):

        # Dataset configurations
        self.root = root
        self.n_mels = n_mels
        self.sr = sr
        self.seconds = seconds
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.n_bits = n_bits
        self.is_validation = is_validation

        # Useful objects
        self.hrtf = HRTF(root)
        if not self.is_validation:
            self.subjects = [
                'subject_135', 'subject_061', 'subject_051', 'subject_012',
                'subject_165', 'subject_137', 'subject_154', 'subject_033',
                'subject_028', 'subject_147', 'subject_156', 'subject_065',
                'subject_152', 'subject_059', 'subject_153', 'subject_019',
                'subject_027', 'subject_048', 'subject_010', 'subject_124',
                'subject_017', 'subject_050', 'subject_040', 'subject_158',
                'subject_163', 'subject_011', 'subject_020', 'subject_015',
                'subject_127', 'subject_162', 'subject_148', 'subject_021'
            ]
        else:
            self.subjects = [
                'subject_058', 'subject_044', 'subject_008', 'subject_119',
                'subject_155', 'subject_126', 'subject_131', 'subject_133'
            ]

        self.wave_fns = [sine_wave, square_wave, triangle_wave]
        
        # Angles
        self.azimuthal = [
            -80, -65, -55, -45, -40, -35, -30, -25, -20,
            -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35,
            40, 45, 55, 65, 80
        ]
        self.elevations = [-45+5.625*x for x in range(50)]
        self.n_azimuthal = len(self.azimuthal)
        self.n_elevations = len(self.elevations)


        # Constants
        self.hop_length = 160   # 10  ms
        self.n_fft = 2048       # 128 ms
        self.win_length = 400   # 25  ms
        self.fmin = 50
        self.top_db = 80

        # Audio operators
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

    def __len__(self):
        return len(self.subjects)*self.n_azimuthal*self.n_elevations
    
    def __getitem__(self, idx):
        # Create a random waveform
        random_freq = np.random.randint(100, 3100)
        wave_fn = np.random.choice(self.wave_fns)
        mono = wave_fn(random_freq, self.sr, self.seconds, self.n_bits)

        az = np.random.randint(0, self.n_azimuthal)
        el = np.random.randint(0, self.n_elevations)

        y = self.get_direction(az, el)

        # Get subject HRIR
        subject = np.random.choice(self.subjects)
        audio = self.hrtf.convolve(mono, self.sr, az, el, subject)
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
            'freq': random_freq,
            'wave_fn': wave_fn.__name__,
            'angles': (az, el),
            'subject': subject
        }

        return X, torch.Tensor([y]), parameters
    
    def get_direction(self, az, el):
        return 0. if self.azimuthal[az] < 0 else 1.
