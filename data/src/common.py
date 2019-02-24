import h5py


def create_dataset(name, images, labels_ami, labels_mse):
    with h5py.File('{}_data.h5'.format(name), 'w') as f:
        for key, val in images.items():
            f.create_dataset(key, data=val, compression='gzip')
    with h5py.File('{}_labels.h5'.format(name), 'w') as f:
        f.create_group('AMI')
        for key, val in labels_ami.items():
            f['AMI'].create_dataset(key, data=val, compression='gzip')
        f.create_group('MSE')
        for key, val in labels_mse.items():
            f['MSE'].create_dataset(key, data=val, compression='gzip')
