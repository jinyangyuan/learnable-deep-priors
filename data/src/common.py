import h5py
import numpy as np


def generate_images(objects):
    images = objects[:, 0, :-1]
    for idx in range(1, objects.shape[1]):
        masks = objects[:, idx, -1:]
        images = images * (1 - masks) + objects[:, idx, :-1] * masks
    return images


def generate_labels_mask(objects, th=0.5):
    masks_rev = objects[:, ::-1, -1]
    part_cumprod = np.concatenate([
        np.ones((masks_rev.shape[0], 1, *masks_rev.shape[2:]), dtype=masks_rev.dtype),
        np.cumprod(1 - masks_rev[:, :-1], 1),
    ], axis=1)
    coef = (masks_rev * part_cumprod)[:, ::-1]
    values = np.argmax(coef, 1).astype(np.int32)
    counts = (masks_rev >= th).sum(1) - 1
    valids = (counts == 1).astype(values.dtype)
    labels = np.stack([values, valids], axis=1)
    return labels


def generate_labels_rgba(objects):
    labels = objects.copy()
    masks = labels[:, 1:, -1:]
    labels[:, 1:, :-1] = labels[:, 1:, :-1] * masks + labels[:, :1, :-1] * (1 - masks)
    return labels


def save_dataset(name, images, labels_mask, labels_rgba):
    with h5py.File('{}_data.h5'.format(name), 'w') as f:
        for key, val in images.items():
            f.create_dataset(key, data=val, compression='gzip', chunks=True)
    with h5py.File('{}_labels.h5'.format(name), 'w') as f:
        f.create_group('mask')
        for key, val in labels_mask.items():
            f['mask'].create_dataset(key, data=val, compression='gzip', chunks=True)
        f.create_group('rgba')
        for key, val in labels_rgba.items():
            f['rgba'].create_dataset(key, data=val, compression='gzip', chunks=True)
    return


def create_dataset(name, objects):
    images = {key: generate_images(val) for key, val in objects.items()}
    labels_mask = {key: generate_labels_mask(val) for key, val in objects.items()}
    labels_rgba = {key: generate_labels_rgba(val) for key, val in objects.items()}
    save_dataset(name, images, labels_mask, labels_rgba)
    return
