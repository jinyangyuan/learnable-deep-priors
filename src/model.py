import math
import torch
import torch.nn as nn
import torch.nn.functional
from network import InitializerBack, InitializerObj, UpdaterBack, UpdaterObj, DecoderBack, DecoderObj


class Model(nn.Module):

    def __init__(self, args):
        super(Model, self).__init__()
        # Hyperparameters
        self.max_objects = args.max_objects
        self.num_steps = args.num_steps
        self.noise_prob = args.noise_prob
        self.gaussian_std = args.gaussian_std
        self.gaussian_invvar = 1 / pow(self.gaussian_std, 2)
        self.gaussian_const = math.log(2 * math.pi / self.gaussian_invvar)
        # Neural networks
        self.init_back = InitializerBack(args)
        self.init_obj = InitializerObj(args)
        self.upd_back = UpdaterBack(args)
        self.upd_obj = UpdaterObj(args)
        self.dec_back = DecoderBack(args)
        self.dec_obj = DecoderObj(args)

    def add_noise(self, images):
        mask = torch.bernoulli(torch.full_like(images, self.noise_prob))
        noise = torch.rand_like(images)
        return mask * noise + (1 - mask) * images

    @staticmethod
    def compute_prob_prior(shp):
        padded_ones = shp.new_ones(1, *shp.shape[1:])
        return torch.cat([shp, padded_ones]) * torch.cat([padded_ones, 1 - shp]).cumprod(0)

    @staticmethod
    def compute_log_prob_prior(logits_shp):
        log_shp = nn.functional.logsigmoid(logits_shp)
        log_one_minus_shp = log_shp - logits_shp
        padded_zeros = log_shp.new_zeros(1, *logits_shp.shape[1:])
        return torch.cat([log_shp, padded_zeros]) + torch.cat([padded_zeros, log_one_minus_shp]).cumsum(0)

    def compute_log_prob_cond(self, images, back, apc):
        diff = images[None] - torch.cat([apc, back[None]])
        return -0.5 * (self.gaussian_const + self.gaussian_invvar * diff.pow(2)).sum(-3, keepdim=True)

    def compute_probabilities(self, images, result_back, result_obj):
        log_prob_prior = self.compute_log_prob_prior(result_obj['logits_shp'])
        log_prob_cond = self.compute_log_prob_cond(images, result_back['back'], result_obj['apc'])
        log_prob = log_prob_prior + log_prob_cond
        gamma = nn.functional.softmax(log_prob, dim=0)
        log_gamma = nn.functional.log_softmax(log_prob, dim=0)
        return {'log_prob': log_prob, 'gamma': gamma, 'log_gamma': log_gamma}

    def compute_init_obj_inputs(self, images, result_back, result_obj):
        prob_prior = self.compute_prob_prior(result_obj['shp'])
        recon = (prob_prior * torch.cat([result_obj['apc'], result_back['back'][None]])).sum(0)
        mask = 1 - prob_prior[-1]
        return torch.cat([images, recon, mask], dim=1).detach()

    @staticmethod
    def compute_upd_back_inputs(noisy_images, result):
        return result['gamma'][-1] * (noisy_images - result['back'])

    @staticmethod
    def compute_upd_obj_inputs(noisy_images, result):
        inputs_apc = result['gamma'][:-1] * (noisy_images[None] - result['apc'])
        inputs_shp = result['gamma'][:-1] * (1 - result['shp']) - (1 - result['gamma'][:-1].cumsum(0)) * result['shp']
        return inputs_apc, inputs_shp

    def initialize_storage(self, result_obj, states_obj, sub_states_obj):
        update_dict = self.dec_obj(sub_states_obj[0], num_objects=1)
        for key, val in result_obj.items():
            result_obj[key] = torch.cat([val, update_dict[key]])
        if states_obj is None:
            states_obj = sub_states_obj
        else:
            states_obj = tuple([torch.cat([n1, n2]) for n1, n2 in zip(states_obj, sub_states_obj)])
        return states_obj

    def compute_indices(self, images, result_obj, eps=1e-5):
        diffs_sq = (result_obj['apc'] - images[None]).pow(2).sum(-3, keepdim=True).detach()
        masks = result_obj['shp'].clone().detach()
        coefs = masks.new_ones(masks.shape[:-2])
        indices_list = []
        for _ in range(diffs_sq.shape[0]):
            vis_diffs_sq = (masks * diffs_sq).view(*masks.shape[:-2], -1).sum(-1)
            vis_areas = masks.view(*masks.shape[:-2], -1).sum(-1)
            vis_max_vals = masks.view(*masks.shape[:-2], -1).max(-1).values
            scores = coefs * vis_max_vals * torch.exp(-0.5 * self.gaussian_invvar * vis_diffs_sq / (vis_areas + eps))
            indices = torch.argmax(scores, dim=0)
            indices_list.append(indices)
            mask = torch.gather(masks, 0, indices[None, ..., None, None].expand(-1, -1, *masks.shape[2:]))
            masks *= 1 - mask
            coefs.scatter_(0, indices[None], -1)
        indices = torch.stack(indices_list)
        return indices

    @staticmethod
    def adjust_order_sub(x, indices):
        if x.dim() == 2:
            x = x.view(indices.shape[0], -1, *x.shape[1:])
            x = torch.gather(x, 0, indices.expand(-1, -1, *x.shape[2:]))
            x = x.view(-1, *x.shape[2:])
        elif x.dim() == 3:
            x = torch.gather(x, 0, indices.expand(-1, -1, *x.shape[2:]))
        elif x.dim() == 5:
            x = torch.gather(x, 0, indices[..., None, None].expand(-1, -1, *x.shape[2:]))
        else:
            raise AssertionError
        return x

    def adjust_order(self, images, result_obj, states_obj):
        indices = self.compute_indices(images, result_obj)
        for key, val in result_obj.items():
            result_obj[key] = self.adjust_order_sub(val, indices)
        states_obj = tuple([self.adjust_order_sub(n, indices) for n in states_obj])
        return states_obj

    @staticmethod
    def transform_result(result):
        for key, val in result.items():
            if key not in ['back', 'back_diff']:
                result[key] = val.transpose(0, 1)
        return result

    def forward(self, images):
        ###################
        # Initializations #
        ###################
        # Background
        states_back = self.init_back(images)
        result_back = self.dec_back(states_back[0])
        # Objects
        result_obj = {
            'apc': images.new_empty(0, *images.shape),
            'apc_diff': images.new_empty(0, *images.shape),
            'shp': images.new_zeros(0, images.shape[0], 1, *images.shape[2:]),
            'logits_shp': images.new_zeros(0, images.shape[0], 1, *images.shape[2:]),
        }
        states_obj, states_main = None, None
        for _ in range(self.max_objects):
            init_obj_inputs = self.compute_init_obj_inputs(images, result_back, result_obj)
            sub_states_obj, states_main = self.init_obj(init_obj_inputs, states_main)
            states_obj = self.initialize_storage(result_obj, states_obj, sub_states_obj)
        states_obj = self.adjust_order(images, result_obj, states_obj)
        # Result
        result_prob = self.compute_probabilities(images, result_back, result_obj)
        result = {**result_back, **result_obj, **result_prob}
        results = [result]
        ###############
        # Refinements #
        ###############
        for _ in range(self.num_steps):
            # Add noises to images
            noisy_images = self.add_noise(images)
            # Compute inputs of the hidden states updaters
            upd_back_inputs = self.compute_upd_back_inputs(noisy_images, result)
            upd_apc_inputs, upd_shp_inputs = self.compute_upd_obj_inputs(noisy_images, result)
            # Update hidden states
            states_back = self.upd_back(upd_back_inputs, states_back)
            states_obj = self.upd_obj(upd_apc_inputs, upd_shp_inputs, states_obj)
            # Decode
            result_back = self.dec_back(states_back[0])
            result_obj = self.dec_obj(states_obj[0], num_objects=self.max_objects)
            # Adjust order
            states_obj = self.adjust_order(images, result_obj, states_obj)
            # Result
            result_prob = self.compute_probabilities(images, result_back, result_obj)
            result = {**result_back, **result_obj, **result_prob}
            results.append(result)
        results = [self.transform_result(n) for n in results]
        return results

    @staticmethod
    def compute_loss_elbo(result):
        log_prob = result['log_prob']
        gamma = result['gamma'].detach()
        log_gamma = result['log_gamma'].detach()
        return (gamma * (log_gamma - log_prob)).sum()

    def compute_loss_back_prior(self, images, back):
        back_prior = images.view(*images.shape[:-2], -1).median(-1).values[..., None, None]
        loss = 0.5 * self.gaussian_invvar * (back - back_prior).pow(2).sum()
        return loss - loss.detach()

    def compute_loss_diff(self, x):
        loss = 0.5 * self.gaussian_invvar * x.pow(2).sum()
        return loss - loss.detach()

    def compute_batch_loss(self, images, result, coef_dict):
        loss_elbo = self.compute_loss_elbo(result)
        loss_back_prior = coef_dict['back_prior'] * self.compute_loss_back_prior(images, result['back'])
        loss_back_diff = coef_dict['back_diff'] * self.compute_loss_diff(result['back_diff'])
        loss_apc_diff = coef_dict['apc_diff'] * self.compute_loss_diff(result['apc_diff'])
        loss = loss_elbo + loss_back_prior + loss_back_diff + loss_apc_diff
        return loss

    def compute_log_likelihood(self, images, result, log_segre, recon_scene):
        diff_mixture = torch.cat([result['apc'], result['back'][None]]) - images[None]
        raw_ll_mixture = -0.5 * (self.gaussian_const + self.gaussian_invvar * diff_mixture.pow(2)).sum(-3, keepdim=True)
        ll_mixture = torch.logsumexp(log_segre + raw_ll_mixture, dim=0)
        ll_mixture = ll_mixture.view(ll_mixture.shape[0], -1).sum(-1)
        diff_single = recon_scene - images
        ll_single = -0.5 * (self.gaussian_const + self.gaussian_invvar * diff_single.pow(2))
        ll_single = ll_single.view(ll_single.shape[0], -1).sum(-1)
        return ll_mixture, ll_single


def get_model(args, path=None):
    model = Model(args)
    if path is not None:
        load_dict = torch.load(path)
        model_dict = model.state_dict()
        for key in model_dict:
            if key in load_dict and model_dict[key].shape == load_dict[key].shape:
                model_dict[key] = load_dict[key]
            else:
                print('"{}" not loaded'.format(key))
        model.load_state_dict(model_dict)
    return nn.DataParallel(model)
