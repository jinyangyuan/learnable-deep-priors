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
        images_prev = {key[:5]: f[key]['features'][()] for key in ['training', 'validation', 'test']}
        labels_mask_prev = {key[:5]: f[key]['groups'][()] for key in ['training', 'validation', 'test']}
        labels_rgba_prev = {key[:5]: f[key]['masks'][()] for key in ['training', 'validation', 'test']}
    images = {key: np.rollaxis(val.squeeze(0), -1, -3) for key, val in images_prev.items()}
    labels_mask_value = {key: np.rollaxis(val.squeeze(0).astype(np.int32), -1, -3)
                         for key, val in labels_mask_prev.items()}
    labels_mask_valid = {key: (val != 0).astype(val.dtype) for key, val in labels_mask_value.items()}
    labels_mask = {key: np.concatenate([labels_mask_value[key], labels_mask_valid[key]], axis=-3)
                   for key in labels_mask_value}
    labels_rgba_value = {key: np.rollaxis(val.squeeze(0), -1, -3)[..., None, :, :]
                         for key, val in labels_rgba_prev.items()}
    labels_rgba_valid = {key: val / val.max() for key, val in labels_rgba_value.items()}
    labels_rgba_obj = {key: np.concatenate([labels_rgba_value[key], labels_rgba_valid[key]], axis=-3)
                       for key in labels_rgba_value}
    labels_rgba_back = {
        key: np.concatenate([
            np.zeros((*val.shape[:-4], 1, val.shape[-3] - 1, *val.shape[-2:]), dtype=val.dtype),
            np.ones((*val.shape[:-4], 1, 1, *val.shape[-2:]), dtype=val.dtype),
        ], axis=-3)
        for key, val in labels_rgba_obj.items()
    }
    labels_rgba = {key: np.concatenate([labels_rgba_back[key], labels_rgba_obj[key]], axis=-4)
                   for key in labels_rgba_obj}
    save_dataset(args.name, images, labels_mask, labels_rgba)
