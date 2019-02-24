import argparse
import h5py
import numpy as np
import os
from common import create_dataset


parser = argparse.ArgumentParser()
parser.add_argument('--folder_downloads')
parser.add_argument('--filename', default='shapes_20x20.h5')
args = parser.parse_args()
with h5py.File(os.path.join(args.folder_downloads, args.filename), 'r') as f:
    images_prev = f['features'][()]
    labels_mse_prev = f['mask'][()]
image_size = 20
images = images_prev.reshape(images_prev.shape[0], 1, image_size, image_size)[:-10000]
labels_mse = labels_mse_prev.reshape(labels_mse_prev.shape[0], -1, 1, image_size, image_size)[:-10000]
labels_mse *= images[:, None]
labels_ami = np.ndarray(images.shape, dtype=images.dtype)
mask = np.zeros(images.shape, dtype=np.int)
for i in range(labels_mse.shape[1]):
    pos_sel = labels_mse[:, i] != 0
    labels_ami[pos_sel] = i + 1
    mask[pos_sel] += 1
labels_ami[mask > 1] = 0
labels_ami = labels_ami.squeeze(1)
sep1, sep2 = 50000, 60000
images = {'train': images[:sep1], 'valid': images[sep1:sep2], 'test': images[sep2:]}
labels_ami = {'train': labels_ami[:sep1], 'valid': labels_ami[sep1:sep2], 'test': labels_ami[sep2:]}
labels_mse = {'train': labels_mse[:sep1], 'valid': labels_mse[sep1:sep2], 'test': labels_mse[sep2:]}
create_dataset('shapes_20x20', images, labels_ami, labels_mse)
