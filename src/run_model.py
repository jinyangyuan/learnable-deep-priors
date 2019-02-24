import h5py
import os
import torch
import torch.optim as optim


def train_model(args, net, dataloader):
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    best_loss = float('inf')
    for epoch in range(args.num_epochs):
        print('Epoch {}/{}'.format(epoch, args.num_epochs - 1))
        print('-' * 10)
        for phase in ['train', 'valid']:
            net.train(phase == 'train')
            sum_loss, num_data = 0, 0
            for images in dataloader[phase]:
                images = images.cuda()
                with torch.set_grad_enabled(phase == 'train'):
                    results = net(images)
                loss, weight = 0, 0
                for result, sub_weight in zip(results, args.loss_weights):
                    sub_loss = net.compute_loss(result)
                    loss = loss + sub_weight * sub_loss
                    weight += sub_weight
                loss /= weight
                optimizer.zero_grad()
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                sum_loss += loss.item() * images.shape[0]
                num_data += images.shape[0]
            mean_loss = sum_loss / num_data
            print('{}\tLoss: {:.4f}'.format(phase.capitalize(), mean_loss))
            if phase == 'valid' and mean_loss < best_loss:
                best_loss = mean_loss
                torch.save(net.state_dict(), os.path.join(args.folder, args.file_model))
        print()
    print('Best Validation\tLoss: {:.4f}'.format(best_loss))


def test_model(args, net, dataloader):
    net.train(False)
    for model_id in range(args.num_tests):
        results_last = {key: [] for key in ['gamma', 'pi', 'mu']}
        for images in dataloader['test']:
            images = images.cuda()
            with torch.set_grad_enabled(False):
                sub_results_last = net(images)[-1]
            for key in results_last:
                result = sub_results_last[key].data.transpose(0, 1)
                if (not args.binary_image) and key == 'mu':
                    result = ((result + 1) * 0.5).clamp_(0, 1)
                results_last[key].append(result.cpu())
        with h5py.File(os.path.join(args.folder, args.file_result_base.format(model_id)), 'w') as f:
            for key, val in results_last.items():
                f.create_dataset(key, data=torch.cat(val).numpy(), compression='gzip')
