import h5py
import numpy as np


def generate_images(objects):
    images = objects[:, 0, :-1]
    for idx in range(1, objects.shape[1]):
        masks = objects[:, idx, -1:]
        images = images * (1 - masks) + objects[:, idx, :-1] * masks
    images = (images * 255).astype(np.uint8)
    return images


def generate_labels(objects, th=0.5):
    masks_rev = objects[:, ::-1, -1]
    part_cumprod = np.concatenate([
        np.ones((masks_rev.shape[0], 1, *masks_rev.shape[2:]), dtype=masks_rev.dtype),
        np.cumprod(1 - masks_rev[:, :-1], 1),
    ], axis=1)
    coef = (masks_rev * part_cumprod)[:, ::-1]
    segments = np.argmax(coef, 1).astype(np.uint8)
    overlaps = ((masks_rev >= th).sum(1) - 1).astype(np.uint8)
    labels = {'segment': segments, 'overlap': overlaps}
    return labels


def save_dataset(name, images, labels, objects):
    with h5py.File('{}.h5'.format(name), 'w') as f:
        for key in images:
            f.create_group(key)
            f[key].create_dataset('image', data=images[key], compression='gzip')
            f[key].create_dataset('segment', data=labels[key]['segment'], compression='gzip')
            f[key].create_dataset('overlap', data=labels[key]['overlap'], compression='gzip')
            f[key].create_dataset('layers', data=objects[key], compression='gzip')
    return


def create_dataset(name, objects):
    images = {key: generate_images(val) for key, val in objects.items()}
    labels = {key: generate_labels(val) for key, val in objects.items()}
    objects = {key: (val * 255).astype(np.uint8) for key, val in objects.items()}
    save_dataset(name, images, labels, objects)
    return
