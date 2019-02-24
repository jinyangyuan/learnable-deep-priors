# Create the Multi MNIST Dataset derived from "Binding via Reconstruction Clustering"
# This is the modified version of https://github.com/Qwlouse/Binding/blob/master/Datasets/Multi-MNIST.ipynb

import argparse
import h5py
import numpy as np
import os
from common import create_dataset


def crop(d):
    return d[np.sum(d, 1) != 0][:, np.sum(d, 0) != 0]


def generate_data(image_size, digit_indices, test=False, binarize_threshold=0.5):
    if not test:
        digits = [crop(mnist_digits[idx].reshape(28, 28)) for idx in digit_indices]
    else:
        digits = [crop(mnist_digits_test[idx].reshape(28, 28)) for idx in digit_indices]
    images = np.zeros((image_size, image_size))
    labels_mse = np.zeros((num_objects, image_size, image_size))
    labels_ami = np.zeros((image_size, image_size))
    overlap_cnt = np.zeros((image_size, image_size))
    for idx, digit in enumerate(digits):
        nrows, ncols = digit.shape
        col = np.random.randint(0, image_size - ncols + 1)
        row = np.random.randint(0, image_size - nrows + 1)
        region = (slice(row, row + nrows), slice(col, col + ncols))
        pos_sel = np.where(digit >= binarize_threshold)
        images[region] = np.max([images[region], digit], axis=0)
        overlap_cnt[region][pos_sel] += 1
        labels_ami[region][pos_sel] = idx + 1
        labels_mse[idx][region][pos_sel] = 1
    pos_valid = overlap_cnt == 1
    labels_ami *= pos_valid
    return images[None], labels_ami, labels_mse[:, None]


parser = argparse.ArgumentParser()
parser.add_argument('--folder_downloads')
args = parser.parse_args()

with h5py.File(os.path.join(args.folder_downloads, 'MNIST.hdf5'), 'r') as f:
    mnist_digits = f['normalized_full/training/default'][0, :]
    mnist_targets = f['normalized_full/training/targets'][:]
    mnist_digits_test = f['normalized_full/test/default'][0, :]
    mnist_targets_test = f['normalized_full/test/targets'][:]

image_size = 48
num_objects = 2
num_data = {'train': 50000, 'valid': 10000, 'test': 10000}
images = {key: np.empty((val, 1, image_size, image_size), dtype=np.float32) for key, val in num_data.items()}
labels_ami = {key: np.empty((val, image_size, image_size), dtype=np.float32) for key, val in num_data.items()}
labels_mse = {key: np.empty((val, num_objects, 1, image_size, image_size), dtype=np.float32)
              for key, val in num_data.items()}

for num_variants in [20, 500]:
    np.random.seed(36520)
    for key in ['train', 'valid', 'test']:
        for idx in range(num_data[key]):
            digit_indices = np.random.randint(0, num_variants, num_objects)
            images[key][idx], labels_ami[key][idx], labels_mse[key][idx] = generate_data(image_size, digit_indices)
    create_dataset('mnist_{}'.format(num_variants), images, labels_ami, labels_mse)

np.random.seed(36520)
for key in ['train', 'valid']:
    for idx in range(num_data[key]):
        digit_indices = np.random.randint(0, 60000, num_objects)
        images[key][idx], labels_ami[key][idx], labels_mse[key][idx] = generate_data(image_size, digit_indices)
key = 'test'
for idx in range(num_data[key]):
    digit_indices = np.random.randint(0, 10000, num_objects)
    images[key][idx], labels_ami[key][idx], labels_mse[key][idx] = generate_data(image_size, digit_indices, test=True)
create_dataset('mnist_all', images, labels_ami, labels_mse)
