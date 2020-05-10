import argparse
import h5py
import numpy as np
import os
from common import save_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='shapes_28x28_3')
    parser.add_argument('--folder_downloads', default='downloads')
    args = parser.parse_args()
    with h5py.File(os.path.join(args.folder_downloads, '{}.h5'.format(args.name)), 'r') as f:
        images_prev = {key[:5]: f[key]['features'][()] for key in f}
        groups_prev = {key[:5]: f[key]['groups'][()] for key in f}
        overlaps_prev = {key[:5]: f[key]['overlaps'][()] for key in f}
        masks_prev = {key[:5]: f[key]['masks'][()] for key in f}
    images, labels, objects = {}, {}, {}
    for key in images_prev:
        images[key] = (np.rollaxis(images_prev[key].squeeze(0), -1, -3) * 255).astype(np.uint8)
        groups = groups_prev[key].squeeze((0, -1)).astype(np.uint8)
        overlaps = overlaps_prev[key].squeeze((0, -1)).astype(np.uint8)
        labels[key] = {'segment': groups, 'overlap': overlaps}
        sub_objects = (np.rollaxis(masks_prev[key].squeeze(0), -1, -3) * 255).astype(np.uint8)
        sub_objects = np.expand_dims(sub_objects, axis=-3).repeat(2, axis=-3)
        objects[key] = np.concatenate(
            [np.zeros([sub_objects.shape[0], 1, *sub_objects.shape[2:]], dtype=np.uint8), sub_objects], axis=1)
    save_dataset(args.name, images, labels, objects)
