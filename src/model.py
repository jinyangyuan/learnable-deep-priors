import torch
import torch.nn as nn
from building_block import Encoder, Decoder, UpdaterRNN, UpdaterLSTM


updater_dict = {'rnn': UpdaterRNN, 'lstm': UpdaterLSTM}


class ModelBase(nn.Module):

    def __init__(self, args):
        super(ModelBase, self).__init__()
        # Hyper-parameters
        self.image_size = args.image_size
        self.num_objects = args.num_objects
        self.num_steps = args.num_steps
        self.noise_prob = args.noise_prob
        self.regularization = args.regularization
        # Hidden states updater
        self.updater = updater_dict[args.update_rule](args.updater_size + args.num_channels, args.state_size)
        # Learning rate of the background component
        self.back_lr = nn.Parameter(torch.FloatTensor([args.back_lr_init]))
        # Encoder and decoder for shape
        self.encoder_pi = Encoder(args.config_conv, 1, args.image_size, args.updater_size, args.normalize_inputs)
        self.decoder_pi = Decoder(args.config_conv, 1, args.image_size, args.updater_size, args.state_size)
        # Decoder for appearance
        self.decoder_mu = nn.Linear(args.state_size, args.num_channels)

    def add_noise(self, images):
        raise NotImplementedError

    def compute_mu(self, outputs_mu):
        raise NotImplementedError

    def compute_log_prob_cond(self, images, outputs_mu):
        raise NotImplementedError

    def compute_loss_kld(self, result):
        raise NotImplementedError

    @staticmethod
    def compute_log_prob_prior(log_pi, logits_pi):
        padded_zeros = log_pi.new_zeros(1, *log_pi.shape[1:])
        log_pi_extend = torch.cat([log_pi, padded_zeros])
        log_one_minus_pi_extend = torch.cat([padded_zeros, log_pi - logits_pi])
        return log_pi_extend + log_one_minus_pi_extend.cumsum(0)

    @staticmethod
    def compute_gamma(log_prob_prior, log_prob_cond):
        log_prob = log_prob_prior + log_prob_cond
        gamma = torch.softmax(log_prob, dim=0)
        return gamma, log_prob

    def compute_updater_inputs_pi(self, gamma, pi):
        gamma_lt_sum = torch.cat([gamma.new_zeros(1, *gamma.shape[1:]), gamma[:-1]]).cumsum(0)
        x = gamma - (1 - gamma_lt_sum) * pi
        x = x.view(-1, *x.shape[2:])
        x = self.encoder_pi(x)
        return x

    @staticmethod
    def compute_updater_inputs_mu(gamma, mu, noisy_images):
        x = gamma * (noisy_images[None] - mu)
        x = x.view(x.shape[0] * x.shape[1], x.shape[2], -1).mean(-1)
        return x

    def decode_pi(self, features):
        logits_pi = self.decoder_pi(features)
        logits_pi = logits_pi.view(self.num_objects, -1, *logits_pi.shape[1:])
        pi = torch.sigmoid(logits_pi)
        log_pi = nn.functional.logsigmoid(logits_pi)
        return pi, log_pi, logits_pi

    def decode_mu(self, features):
        outputs_mu = self.decoder_mu(features)
        outputs_mu = outputs_mu.view(self.num_objects, -1, outputs_mu.shape[-1], 1, 1)
        return outputs_mu

    def forward(self, images):
        # Initializations
        features, states = self.updater.init_states(self.num_objects * images.shape[0])
        pi, log_pi, logits_pi = self.decode_pi(features)
        outputs_mu_obj = self.decode_mu(features)
        outputs_mu_back = self.outputs_mu_prior.expand(-1, images.shape[0], -1, -1, -1)
        outputs_mu = torch.cat([outputs_mu_obj, outputs_mu_back])
        mu = self.compute_mu(outputs_mu)
        log_prob_prior = self.compute_log_prob_prior(log_pi, logits_pi)
        log_prob_cond = self.compute_log_prob_cond(images, outputs_mu)
        gamma, log_prob = self.compute_gamma(log_prob_prior, log_prob_cond)
        results = []
        # Iterations
        for _ in range(self.num_steps):
            # Add noises to images
            noisy_images = self.add_noise(images)
            # Compute inputs of the hidden states updater and the background updater
            delta_pi = self.compute_updater_inputs_pi(gamma[:-1], pi)
            delta_mu_obj = self.compute_updater_inputs_mu(gamma[:-1], mu[:-1], noisy_images)
            delta_mu_back = self.compute_updater_inputs_mu(gamma[-1:], mu[-1:], noisy_images)
            # Update hidden states
            updater_inputs = torch.cat([delta_pi, self.image_size * self.image_size * delta_mu_obj], dim=-1)
            features, states = self.updater(updater_inputs, states)
            # Decode
            pi, log_pi, logits_pi = self.decode_pi(features)
            outputs_mu_obj = self.decode_mu(features)
            outputs_mu_back = outputs_mu_back + self.back_lr * delta_mu_back.view(outputs_mu_back.shape)
            outputs_mu = torch.cat([outputs_mu_obj, outputs_mu_back])
            mu = self.compute_mu(outputs_mu)
            # Compute probabilities
            log_prob_prior = self.compute_log_prob_prior(log_pi, logits_pi)
            log_prob_cond = self.compute_log_prob_cond(images, outputs_mu)
            gamma, log_prob = self.compute_gamma(log_prob_prior, log_prob_cond)
            # Save results
            result = {'gamma': gamma, 'log_prob': log_prob, 'pi': pi, 'mu': mu, 'outputs_mu': outputs_mu}
            results.append(result)
        return results

    def compute_loss(self, result):
        gamma = result['gamma'].detach()
        log_prob = result['log_prob']
        loss_base = -(gamma * log_prob).mean() * (self.num_objects + 1)
        loss_kld = self.compute_loss_kld(result)
        return loss_base + self.regularization * loss_kld


class ModelBinary(ModelBase):

    def __init__(self, args):
        super(ModelBinary, self).__init__(args)
        self.register_buffer('mu_prior', torch.FloatTensor(args.mu_prior).view(1, 1, -1, 1, 1))
        self.register_buffer('outputs_mu_prior', (self.mu_prior.log() - (1 - self.mu_prior).log()).clamp_(-1000, 1000))

    def add_noise(self, images):
        mask = torch.bernoulli(torch.full_like(images, self.noise_prob))
        return images + mask - 2 * images * mask

    def compute_mu(self, logits_mu):
        return torch.sigmoid(logits_mu)

    def compute_log_prob_cond(self, images, logits_mu):
        log_mu = nn.functional.logsigmoid(logits_mu)
        return (log_mu + (images[None] - 1) * logits_mu).sum(-3, keepdim=True)

    def compute_loss_kld(self, result):
        logits_mu = result['outputs_mu']
        log_mu = nn.functional.logsigmoid(logits_mu)
        loss_kld1 = -(log_mu[:-1] - self.mu_prior * logits_mu[:-1]).sum(-3, keepdim=True).mean()
        loss_kld2 = -(log_mu[-1:] + (self.mu_prior - 1) * logits_mu[-1:]).sum(-3, keepdim=True).mean()
        return loss_kld1 + loss_kld2


class ModelReal(ModelBase):

    def __init__(self, args):
        super(ModelReal, self).__init__(args)
        self.register_buffer('mu_prior', torch.FloatTensor(args.mu_prior).view(1, 1, -1, 1, 1) * 2 - 1)
        self.register_buffer('outputs_mu_prior', self.mu_prior)
        self.inv_var = 0.25 / pow(args.gaussian_std, 2)

    def add_noise(self, images):
        mask = torch.bernoulli(torch.full_like(images, self.noise_prob))
        noise = torch.empty_like(images).uniform_(-1, 1)
        return mask * noise + (1 - mask) * images

    def compute_mu(self, mu):
        return mu

    def compute_log_prob_cond(self, images, mu):
        return -0.5 * self.inv_var * (images[None] - mu).pow(2).sum(-3, keepdim=True)

    def compute_loss_kld(self, result):
        mu_back = result['mu'][-1:]
        return 0.5 * self.inv_var * (self.mu_prior - mu_back).pow(2).sum(-3, keepdim=True).mean()


def get_model(args, path=None):
    model = ModelBinary(args) if args.binary_image else ModelReal(args)
    if path is not None:
        model.load_state_dict(torch.load(path))
    return model
