import h5py
import torch
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):

    def __init__(self, args, data):
        self.data = data if args.binary_image else data * 2 - 1

    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx])

    def __len__(self):
        return self.data.shape[0]


def get_dataloader(args):
    with h5py.File(args.path_data, 'r', libver='latest', swmr=True) as f:
        data = {key: f[key][()] for key in f}
    dataset = {key: CustomDataset(args, val) for key, val in data.items()}
    dataloader = {key: DataLoader(val, batch_size=args.batch_size, num_workers=args.num_workers,
                                  shuffle=(key == 'train'), drop_last=(key == 'train')) for key, val in dataset.items()}
    return dataloader
