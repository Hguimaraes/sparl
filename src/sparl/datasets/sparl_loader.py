from torch.utils import data
from sparl.datasets.sparl_dataset import SPARL

def sparl_loader(root, batch_size, shuffle, num_workers, 
        n_mels=128, min_freq=100, max_freq=3100, sr=16000,
        seconds=.5, n_bits=8, is_validation=False):
    # Create the dataset and return a dataloader associated
    dataset = SPARL(
        root=root, 
        n_mels=n_mels, 
        min_freq=min_freq, 
        max_freq=max_freq, 
        sr=sr, 
        seconds=seconds,
        n_bits=n_bits,
        is_validation=is_validation
    )

    return data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )