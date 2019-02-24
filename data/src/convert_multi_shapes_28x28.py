import argparse
import h5py
import os
from common import create_dataset


parser = argparse.ArgumentParser()
parser.add_argument('--folder_downloads')
parser.add_argument('--filename', default='shapes_28x28.h5')
args = parser.parse_args()
with h5py.File(os.path.join(args.folder_downloads, args.filename), 'r') as f:
    images_prev = {key[:5]: f[key]['features'][()] for key in ['training', 'validation', 'test']}
    labels_ami_prev = {key[:5]: f[key]['groups'][()] for key in ['training', 'validation', 'test']}
    labels_mse_prev = {key[:5]: f[key]['masks'][()] for key in ['training', 'validation', 'test']}
images = {key: val[0, :, None, :, :, 0] for key, val in images_prev.items()}
labels_ami = {key: val[0, :, :, :, 0] for key, val in labels_ami_prev.items()}
labels_mse = {key: val.transpose(1, 4, 0, 2, 3) for key, val in labels_mse_prev.items()}
create_dataset('shapes_28x28_3', images, labels_ami, labels_mse)
