import torch
import pathlib
import itertools
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from monai.transforms import LoadImage
from utils.data_util import get_transform

LOAD = LoadImage(image_only=True,ensure_channel_first=True)

class Lung(Dataset):
    def __init__(self, base_dir=None, split='train', label_transform=None, unlabel_transform=None):
        self._base_dir = pathlib.Path(base_dir)
        self.image_list = []
        self.label_transform = label_transform
        self.unlabel_transform = unlabel_transform
        self.split = split

        if split=='train':
            self.image_list = [dir for dir in (self._base_dir/'train').iterdir() if dir.is_dir()]
        elif split == 'test':
            self.image_list = [dir for dir in (self._base_dir/'test').iterdir() if dir.is_dir()]

        self.image_list = sorted(self.image_list)

        print("total {} samples".format(len(self.image_list)))
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        dir_name = self.image_list[idx]
        img = dir_name/(dir_name.name+'.nii.gz')
        label = dir_name/(dir_name.name+'_label.nii.gz')
        img = LOAD(img)

        if label.exists():
            label = LOAD(label)
        else:
            label = torch.zeros_like(img)
        
        sample = {'image': img, 'label': label.long()}

        if idx < 11 and idx >=0 and self.split=='train':
            if self.label_transform :
                sample = self.label_transform(sample)
            if isinstance(sample, (list, tuple)):
                sample = sample[0]
        if idx >= 11 and idx < 54 and self.split=='train':
            if self.unlabel_transform:
                sample = self.unlabel_transform(sample)
            if isinstance(sample, (list, tuple)):
                sample = sample[0]

        sample = {k: v for k, v in sample.items() if k in ('image', 'label')}
        
        label = sample['label'].squeeze(0)
        sample['label'] = label

        return sample
    
class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


if __name__ == "__main__":

    label_transform, unlabel_transform = get_transform()
    lung = Lung(base_dir='D:/pythonProject/seg/data/lung', split='train', label_transform=label_transform, unlabel_transform=unlabel_transform)
    sample = lung[9]
    image = sample['image'][0][:,:,20]
    label = sample['label'][:,:,20]

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2)
    axes[0].imshow(image, cmap='gray')
    axes[1].imshow(label, cmap='gray')
    fig.savefig('lung.png')
    plt.close()


        

    