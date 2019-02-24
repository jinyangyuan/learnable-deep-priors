# The code for plotting results are based on https://github.com/sjoerdvansteenkiste/Neural-EM/blob/master/utils.py

import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import hsv_to_rgb
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_mutual_info_score


def load_dataset(folder, name):
    with h5py.File(os.path.join(folder, '{}_data.h5'.format(name)), 'r') as f:
        images = f['test'][()]
    with h5py.File(os.path.join(folder, '{}_labels.h5'.format(name)), 'r') as f:
        labels_ami = f['AMI']['test'][()]
        labels_mse = f['MSE']['test'][()]
    return images, labels_ami, labels_mse


def load_results(folder, filename='result.h5'):
    with h5py.File(os.path.join(folder, filename), 'r') as f:
        results = {key: f[key][()] for key in f}
    return results


def convert_results(results):
    gamma = results['gamma']
    pi = results['pi']
    mu = results['mu']
    predictions = gamma[:, :-1].argmax(1).squeeze(1)
    reconstructions = pi * mu[:, :-1] + (1 - pi) * mu[:, -1:]
    return gamma, predictions, reconstructions


def compute_ami_score(predictions, labels_ami):
    scores = []
    for prediction, label in zip(predictions, labels_ami):
        pos_sel = np.where(label != 0)
        scores.append(adjusted_mutual_info_score(prediction[pos_sel], label[pos_sel]))
    return np.mean(scores)


def compute_mse_score(reconstructions, labels_mse, valid_all=True):
    valids = labels_mse.prod(axis=1) == 0
    scores = []
    for reconstruction, label, valid in zip(reconstructions, labels_mse, valids):
        cost = np.ndarray((reconstruction.shape[0], label.shape[0]))
        for i in range(reconstruction.shape[0]):
            for j in range(label.shape[0]):
                if valid_all:
                    cost[i, j] = np.square(reconstruction[i] - label[j]).mean()
                else:
                    cost[i, j] = np.square(reconstruction[i][valid] - label[j][valid]).mean()
        rows, cols = linear_sum_assignment(cost)
        scores.append(cost[rows, cols])
    return np.mean(scores)


def compute_gamma_combine(gamma, num_colors):
    hsv_colors = np.ones((num_colors, 3))
    hsv_colors[:, 0] = (np.linspace(0, 1, num_colors, endpoint=False) + 2 / 3) % 1.0
    gamma_colors = hsv_to_rgb(hsv_colors)
    gamma_combine = np.clip((gamma * gamma_colors[None, ..., None, None]).sum(1), 0, 1)
    return gamma_combine, gamma_colors


def convert_image(img):
    img = (np.transpose(img, [1, 2, 0]) * 255).astype(np.uint8)
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    return img


def color_spines(ax, color, lw=3):
    for loc in ['top', 'bottom', 'left', 'right']:
        ax.spines[loc].set_linewidth(lw)
        ax.spines[loc].set_color(color)
        ax.spines[loc].set_visible(True)


def plot_image(ax, img, xlabel=None, ylabel=None, border_color=None):
    ax.imshow(img, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(xlabel, color='k') if xlabel else None
    ax.set_ylabel(ylabel, color='k') if ylabel else None
    ax.xaxis.set_label_position('top')
    if border_color:
        color_spines(ax, color=border_color)


def plot_samples(images, gamma, reconstructions, num_images=15):
    gamma_combine, gamma_colors = compute_gamma_combine(gamma, gamma.shape[1])
    nrows, ncols = reconstructions.shape[1] + 2, num_images
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols, nrows))
    for idx in range(num_images):
        plot_image(axes[0, idx], convert_image(images[idx]), ylabel='scene' if idx == 0 else None)
        plot_image(axes[1, idx], convert_image(gamma_combine[idx]), ylabel='$\gamma$' if idx == 0 else None)
        for idx_sub in range(reconstructions.shape[1]):
            plot_image(axes[idx_sub + 2, idx], convert_image(reconstructions[idx, idx_sub]),
                       ylabel='obj {}'.format(idx_sub + 1) if idx == 0 else None,
                       border_color=tuple(gamma_colors[idx_sub]))
    plt.subplots_adjust(hspace=0.1, wspace=0.1)
