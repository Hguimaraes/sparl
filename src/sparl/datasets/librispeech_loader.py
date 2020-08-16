from torch.utils import data
from sparl.datasets.librispeech_dataset import LibriSpeech

def librispeech_loader(
    batch_size, shuffle, num_workers,
    ls_root, url, cipic_root, n_mels=128, seconds=.5, n_bits=8):

    # Create the dataset and return a dataloader associated
    dataset = LibriSpeech(
        ls_root=ls_root,
        url=url,
        cipic_root=cipic_root,
        n_mels=n_mels, 
        seconds=seconds,
        n_bits=n_bits
    )

    return data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )