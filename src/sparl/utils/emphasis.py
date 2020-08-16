import torch
import torch.nn as nn
from scipy.signal import lfilter

"""
Pre-emphasis filter
Based on: https://github.com/bshall/UniversalVocoding/blob/master/utils.py
"""
class PreEmphasis(nn.Module):
    def __init__(self, alpha=.85):
        super(PreEmphasis, self).__init__()
        self.alpha = alpha
    
    def forward(self, signal):
        emph = lfilter([1, -self.alpha], [1], signal)
        return torch.Tensor(emph)


"""
De-emphasis filter
Based on: https://github.com/bshall/UniversalVocoding/blob/master/utils.py
"""
class DeEmphasis(nn.Module):
    def __init__(self, alpha=.85):
        super(DeEmphasis, self).__init__()
        self.alpha = alpha
    
    def forward(self, signal):
        deemph = lfilter([1], [1, -self.alpha], signal)
        return torch.Tensor(deemph)
