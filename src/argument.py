import argparse


def get_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu')
    parser.add_argument('--dataset')
    parser.add_argument('--path_data')
    parser.add_argument('--folder')
    parser.add_argument('--train', type=int)
    parser.add_argument('--state_size', type=int)
    parser.add_argument('--updater_size', type=int)
    parser.add_argument('--file_model', default='model.pth')
    parser.add_argument('--file_result_base', default='result_{}.h5')
    parser.add_argument('--update_rule', default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--num_tests', type=int, default=5)
    parser.add_argument('--num_steps', type=int, default=15)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--num_channels', type=int, default=1)
    parser.add_argument('--num_objects', type=int, default=None)
    parser.add_argument('--image_size', type=int, default=None)
    parser.add_argument('--normalize_inputs', type=int, default=None)
    parser.add_argument('--binary_image', type=int, default=None)
    parser.add_argument('--config_conv', type=int, nargs='+', default=None)
    parser.add_argument('--noise_prob', type=float, default=0.2)
    parser.add_argument('--gaussian_std', type=float, default=0.25)
    parser.add_argument('--regularization', type=float, default=10.)
    parser.add_argument('--back_lr_init', type=float, default=1e-5)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--mu_prior', type=float, nargs='+', default=[0.])
    parser.add_argument('--loss_weights', type=float, nargs='+', default=None)
    args = parser.parse_args()
    args.loss_weights = args.loss_weights if args.loss_weights is not None else [1.] * args.num_steps
    if args.dataset == 'shapes_20x20':
        args.update_rule = 'lstm'
        args.num_objects = 3
        args.image_size = 20
        args.normalize_inputs = 0
        args.binary_image = 1
        args.config_conv = [32, 64]
        args.lr = 4e-4
    elif args.dataset == 'shapes_28x28':
        args.update_rule = 'lstm'
        args.num_objects = args.num_objects if args.num_objects is not None else 3
        args.image_size = 28
        args.normalize_inputs = 0
        args.binary_image = args.binary_image if args.binary_image is not None else 1
        args.config_conv = [32, 64]
        args.lr = 1e-3
    elif args.dataset == 'mnist':
        args.update_rule = 'rnn'
        args.num_objects = 2
        args.image_size = 48
        args.normalize_inputs = 1
        args.binary_image = 0
        args.config_conv = [32, 64, 128]
        args.lr = 1e-4
    else:
        raise Exception('"dataset" should be in ["shapes_20x20", "shapes_28x28", "mnist"]')
    return args
