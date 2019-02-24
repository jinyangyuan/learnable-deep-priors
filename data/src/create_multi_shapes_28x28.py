# Create the Multi Shapes Dataset derived from "Binding via Reconstruction Clustering"
# This is the modified version of https://github.com/Qwlouse/Binding/blob/master/Datasets/Shapes.ipynb

import argparse
import numpy as np
from common import create_dataset


def generate_data(image_size, num_objects):
    images = np.zeros((image_size, image_size))
    labels_mse = np.zeros((num_objects, image_size, image_size))
    for idx in range(num_objects):
        obj_id = np.random.randint(0, len(shapes))
        shape = shapes[obj_id]
        nrows, ncols = shape.shape
        col = np.random.randint(0, image_size - ncols + 1)
        row = np.random.randint(0, image_size - nrows + 1)
        region = (slice(row, row + nrows), slice(col, col + ncols))
        images[region] = np.max([images[region], shape], axis=0)
        labels_mse[idx][region] = shape
    labels_ami = np.argmax(labels_mse, axis=0) + 1
    pos_valid = labels_mse.sum(0) == 1
    labels_ami *= pos_valid
    return images[None], labels_ami, labels_mse[:, None]


parser = argparse.ArgumentParser()
parser.add_argument('--num_objects', type=int)
args = parser.parse_args()

square = np.array(
    [[1, 1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1, 1],
     [1, 1, 0, 0, 0, 1, 1],
     [1, 1, 0, 0, 0, 1, 1],
     [1, 1, 0, 0, 0, 1, 1],
     [1, 1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1, 1]])

triangle = np.array(
    [[0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
     [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
     [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
     [0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0],
     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

shapes = [square, triangle, triangle[::-1, :].copy()]

image_size = 28
num_objects = args.num_objects
num_data = {'train': 50000, 'valid': 10000, 'test': 10000}
images = {key: np.empty((val, 1, image_size, image_size), dtype=np.float32) for key, val in num_data.items()}
labels_ami = {key: np.empty((val, image_size, image_size), dtype=np.float32) for key, val in num_data.items()}
labels_mse = {key: np.empty((val, num_objects, 1, image_size, image_size), dtype=np.float32)
              for key, val in num_data.items()}

np.random.seed(265076)
for key in ['train', 'valid', 'test']:
    for idx in range(num_data[key]):
        images[key][idx], labels_ami[key][idx], labels_mse[key][idx] = generate_data(image_size, num_objects)
create_dataset('shapes_28x28_{}'.format(num_objects), images, labels_ami, labels_mse)
