import math
import torch
import torch.nn as nn
import torch.nn.functional as nn_func
from network import InitializerBck, InitializerObj, UpdaterBck, UpdaterObj, DecoderBck, DecoderObj


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        # Hyperparameters
        self.obj_slots = config['num_slots'] - 1
        self.num_steps = config['num_steps']
        self.noise_prob = config['noise_prob']
        self.normal_invvar = 1 / pow(config['normal_scale'], 2)
        self.normal_const = math.log(2 * math.pi / self.normal_invvar)
        self.seg_bck = config['seg_bck']
        # Neural networks
        self.init_bck = InitializerBck(config)
        self.init_obj = InitializerObj(config)
        self.upd_bck = UpdaterBck(config)
        self.upd_obj = UpdaterObj(config)
        self.dec_bck = DecoderBck(config)
        self.dec_obj = DecoderObj(config)

    def forward(self, images, labels, step_wt):
        ###################
        # Initializations #
        ###################
        # Background
        states_bck = self.init_bck(images)
        result_bck = self.dec_bck(states_bck[0])
        # Objects
        result_obj = {
            'apc': torch.zeros([0, *images.shape], device=images.device),
            'apc_res': torch.zeros([0, *images.shape], device=images.device),
            'shp': torch.zeros([0, images.shape[0], 1, *images.shape[2:]], device=images.device),
            'logits_shp': torch.zeros([0, images.shape[0], 1, *images.shape[2:]], device=images.device),
        }
        states_main, states_obj = None, None
        for _ in range(self.obj_slots):
            init_obj_inputs = self.compute_init_obj_in(images, result_bck, result_obj)
            sub_states_obj, states_main = self.init_obj(init_obj_inputs, states_main)
            states_obj = self.initialize_storage(result_obj, states_obj, sub_states_obj)
        states_obj = self.adjust_order(images, result_obj, states_obj)
        # Losses
        result_prob = self.compute_probabilities(images, result_bck, result_obj)
        step_losses = self.compute_step_losses(images, result_bck, result_obj, result_prob)
        losses = {key: [val] for key, val in step_losses.items()}
        ###############
        # Refinements #
        ###############
        for _ in range(self.num_steps):
            # Add noises to images
            noisy_images = self.add_noise(images)
            # Compute inputs of the hidden states updaters
            upd_bck_in, upd_apc_in, upd_shp_in = self.compute_upd_in(noisy_images, result_bck, result_obj, result_prob)
            # Update hidden states
            states_bck = self.upd_bck(upd_bck_in, states_bck)
            states_obj = self.upd_obj(upd_apc_in, upd_shp_in, states_obj)
            # Decode
            result_bck = self.dec_bck(states_bck[0])
            result_obj = self.dec_obj(states_obj[0], obj_slots=self.obj_slots)
            # Adjust order
            states_obj = self.adjust_order(images, result_obj, states_obj)
            # Losses
            result_prob = self.compute_probabilities(images, result_bck, result_obj)
            step_losses = self.compute_step_losses(images, result_bck, result_obj, result_prob)
            for key, val in step_losses.items():
                losses[key].append(val)
        ###########
        # Outputs #
        ###########
        # Losses
        sum_step_wt = step_wt.sum(1)
        losses = {key: torch.stack([loss for loss in val], dim=1) for key, val in losses.items()}
        losses = {key: (step_wt * val).sum(1) / sum_step_wt for key, val in losses.items()}
        # Results
        apc_all = torch.cat([result_obj['apc'], result_bck['bck'][None]]).transpose(0, 1)
        shp = result_obj['shp']
        shp_all = torch.cat([shp, torch.ones([1, *shp.shape[1:]], device=shp.device)]).transpose(0, 1)
        mask = self.compute_mask(shp).transpose(0, 1)
        recon = (mask * apc_all).sum(1)
        segment_all = torch.argmax(mask, dim=1, keepdim=True)
        segment_obj = torch.argmax(mask[:, :-1], dim=1, keepdim=True)
        mask_oh_all = torch.zeros_like(mask)
        mask_oh_obj = torch.zeros_like(mask[:, :-1])
        mask_oh_all.scatter_(1, segment_all, 1)
        mask_oh_obj.scatter_(1, segment_obj, 1)
        pres_all = mask_oh_all.reshape(*mask_oh_all.shape[:-3], -1).max(-1).values
        pres = mask_oh_obj.reshape(*mask_oh_obj.shape[:-3], -1).max(-1).values
        results = {'apc': apc_all, 'shp': shp_all, 'pres': pres_all, 'recon': recon, 'mask': mask,
                   'segment_all': segment_all, 'segment_obj': segment_obj}
        # Metrics
        metrics = self.compute_metrics(images, labels, pres, mask_oh_all, mask_oh_obj, recon, result_prob['log_prob'])
        losses['compare'] = -metrics['ll']
        return results, metrics, losses

    def add_noise(self, images):
        noise_mask = torch.bernoulli(
            torch.full([images.shape[0], 1, *images.shape[2:]], self.noise_prob, device=images.device))
        noise_value = torch.rand_like(images)
        return noise_mask * noise_value + (1 - noise_mask) * images

    def compute_probabilities(self, images, result_bck, result_obj):
        def compute_log_mask(logits_shp):
            log_shp = nn_func.logsigmoid(logits_shp)
            log1m_shp = log_shp - logits_shp
            zeros = torch.ones([1, *logits_shp.shape[1:]], device=logits_shp.device)
            return torch.cat([log_shp, zeros]) + torch.cat([zeros, log1m_shp]).cumsum(0)
        def compute_raw_pixel_ll(bck, apc):
            diff = torch.cat([apc, bck[None]]) - images[None]
            return -0.5 * (self.normal_const + self.normal_invvar * diff.pow(2)).sum(-3, keepdim=True)
        log_mask = compute_log_mask(result_obj['logits_shp'])
        raw_pixel_ll = compute_raw_pixel_ll(result_bck['bck'], result_obj['apc'])
        log_prob = log_mask + raw_pixel_ll
        gamma = nn_func.softmax(log_prob, dim=0)
        log_gamma = nn_func.log_softmax(log_prob, dim=0)
        return {'log_prob': log_prob, 'gamma': gamma, 'log_gamma': log_gamma}

    @staticmethod
    def compute_mask(shp):
        ones = torch.ones([1, *shp.shape[1:]], device=shp.device)
        return torch.cat([shp, ones]) * torch.cat([ones, 1 - shp]).cumprod(0)

    def compute_init_obj_in(self, images, result_bck, result_obj):
        mask = self.compute_mask(result_obj['shp'])
        recon = (mask * torch.cat([result_obj['apc'], result_bck['bck'][None]])).sum(0)
        return torch.cat([images, recon, mask[-1]], dim=1).detach()

    @staticmethod
    def compute_upd_in(noisy_images, result_bck, result_obj, result_prob):
        upd_bck_in = result_prob['gamma'][-1] * (noisy_images - result_bck['bck'])
        upd_apc_in = result_prob['gamma'][:-1] * (noisy_images[None] - result_obj['apc'])
        upd_shp_in = result_prob['gamma'][:-1] * (1 - result_obj['shp']) \
                     - (1 - result_prob['gamma'][:-1].cumsum(0)) * result_obj['shp']
        return upd_bck_in, upd_apc_in, upd_shp_in

    def initialize_storage(self, result_obj, states_obj, sub_states_obj):
        update_dict = self.dec_obj(sub_states_obj[0], obj_slots=1)
        for key, val in result_obj.items():
            result_obj[key] = torch.cat([val, update_dict[key]])
        if states_obj is None:
            states_obj = sub_states_obj
        else:
            states_obj = tuple([torch.cat([n1, n2]) for n1, n2 in zip(states_obj, sub_states_obj)])
        return states_obj

    def adjust_order(self, images, result_obj, states_obj, eps=1e-5):
        def permute(x):
            if x.dim() == 3:
                indices_expand = indices
            else:
                indices_expand = indices[..., None, None]
            x = torch.gather(x, 0, indices_expand.expand(-1, -1, *x.shape[2:]))
            return x
        sq_diffs = (result_obj['apc'] - images[None]).pow(2).sum(-3, keepdim=True).detach()
        visibles = result_obj['shp'].clone().detach()
        coefs = torch.ones(visibles.shape[:-2], device=visibles.device)
        indices_list = []
        for _ in range(self.obj_slots):
            vis_sq_diffs = (visibles * sq_diffs).sum([-2, -1])
            vis_areas = visibles.sum([-2, -1])
            vis_max_vals = visibles.reshape(*visibles.shape[:-2], -1).max(-1).values
            scores = torch.exp(-0.5 * self.normal_invvar * vis_sq_diffs / (vis_areas + eps))
            scaled_scores = coefs * (vis_max_vals * scores + 1)
            indices = torch.argmax(scaled_scores, dim=0, keepdim=True)
            indices_list.append(indices)
            vis = torch.gather(visibles, 0, indices[..., None, None].expand(-1, -1, *visibles.shape[2:]))
            visibles *= 1 - vis
            coefs.scatter_(0, indices, -1)
        indices = torch.cat(indices_list)
        for key, val in result_obj.items():
            result_obj[key] = permute(val)
        states_obj = [n.reshape(indices.shape[0], -1, *n.shape[1:]) for n in states_obj]
        states_obj = [permute(n) for n in states_obj]
        states_obj = [n.reshape(-1, *n.shape[2:]) for n in states_obj]
        return tuple(states_obj)

    def compute_step_losses(self, images, result_bck, result_obj, result_prob):
        # Loss ELBO
        log_prob = result_prob['log_prob']
        gamma = result_prob['gamma'].detach()
        log_gamma = result_prob['log_gamma'].detach()
        loss_elbo = (gamma * (log_gamma - log_prob)).sum([0, *range(2, gamma.dim())])
        # Loss back prior
        bck_prior = images.reshape(*images.shape[:-2], -1).median(-1).values[..., None, None]
        sq_diff = (result_bck['bck'] - bck_prior).pow(2)
        loss_bck_prior = 0.5 * self.normal_invvar * sq_diff.sum([*range(1, sq_diff.dim())])
        # Loss back variance
        bck_var = result_bck['bck_res'].pow(2)
        loss_bck_var = 0.5 * self.normal_invvar * bck_var.sum([*range(1, bck_var.dim())])
        # Loss apc variance
        apc_var = result_obj['apc_res'].pow(2)
        loss_apc_var = 0.5 * self.normal_invvar * apc_var.sum([0, *range(2, apc_var.dim())])
        # Losses
        losses = {'elbo': loss_elbo, 'bck_prior': loss_bck_prior, 'bck_var': loss_bck_var, 'apc_var': loss_apc_var}
        return losses

    @staticmethod
    def compute_ari(mask_true, mask_pred):
        def comb2(x):
            x = x * (x - 1)
            if x.dim() > 1:
                x = x.sum([*range(1, x.dim())])
            return x
        num_pixels = mask_true.sum([*range(1, mask_true.dim())])
        mask_true = mask_true.reshape(
            [mask_true.shape[0], mask_true.shape[1], 1, mask_true.shape[-2] * mask_true.shape[-1]])
        mask_pred = mask_pred.reshape(
            [mask_pred.shape[0], 1, mask_pred.shape[1], mask_pred.shape[-2] * mask_pred.shape[-1]])
        mat = (mask_true * mask_pred).sum(-1)
        sum_row = mat.sum(1)
        sum_col = mat.sum(2)
        comb_mat = comb2(mat)
        comb_row = comb2(sum_row)
        comb_col = comb2(sum_col)
        comb_num = comb2(num_pixels)
        comb_prod = (comb_row * comb_col) / comb_num
        comb_mean = 0.5 * (comb_row + comb_col)
        diff = comb_mean - comb_prod
        score = (comb_mat - comb_prod) / diff
        invalid = ((comb_num == 0) + (diff == 0)) > 0
        score = torch.where(invalid, torch.ones_like(score), score)
        return score

    def compute_metrics(self, images, labels, pres, mask_oh_all, mask_oh_obj, recon, log_prob):
        # ARI
        ari_all = self.compute_ari(labels, mask_oh_all)
        ari_obj = self.compute_ari(labels, mask_oh_obj)
        # MSE
        sq_diff = (recon - images).pow(2)
        mse = sq_diff.mean([*range(1, sq_diff.dim())])
        # Log-likelihood
        pixel_ll = torch.logsumexp(log_prob, dim=0)
        ll = pixel_ll.sum([*range(1, pixel_ll.dim())])
        # Count
        pres_true = labels.reshape(*labels.shape[:-3], -1).max(-1).values
        if self.seg_bck:
            pres_true = pres_true[:, 1:]
        count_true = pres_true.sum(1)
        count_pred = pres.sum(1)
        count_acc = (count_true == count_pred).to(dtype=torch.float)
        metrics = {'ari_all': ari_all, 'ari_obj': ari_obj, 'mse': mse, 'll': ll, 'count': count_acc}
        return metrics

    @staticmethod
    def compute_overview(images, results):
        def convert_single(x_in, color=None):
            x = nn_func.pad(x_in, [boarder_size] * 4, value=0)
            if color is not None:
                boarder = nn_func.pad(torch.zeros_like(x_in), [boarder_size] * 4, value=1) * color
                x += boarder
            x = nn_func.pad(x, [boarder_size] * 4, value=1)
            return x
        def convert_multiple(x, color=None):
            batch_size, num_slots = x.shape[:2]
            x = x.reshape(batch_size * num_slots, *x.shape[2:])
            if color is not None:
                color = color.reshape(batch_size * num_slots, *color.shape[2:])
            x = convert_single(x, color=color)
            x = x.reshape(batch_size, num_slots, *x.shape[1:])
            x = torch.cat(torch.unbind(x, dim=1), dim=-1)
            return x
        boarder_size = round(min(images.shape[-2:]) / 32)
        images = images.expand(-1, 3, -1, -1)
        recon = results['recon'].expand(-1, 3, -1, -1)
        apc = results['apc'].expand(-1, -1, 3, -1, -1)
        shp = results['shp'].expand(-1, -1, 3, -1, -1)
        pres = results['pres'][..., None, None, None].expand(-1, -1, 3, -1, -1)
        color_0 = torch.zeros_like(pres)
        color_1 = torch.zeros_like(pres)
        color_0[..., 1, :, :] = 0.5
        color_0[..., 2, :, :] = 1
        color_1[..., 0, :, :] = 1
        color_1[..., 1, :, :] = 0.5
        boarder_color = pres * color_1 + (1 - pres) * color_0
        boarder_color[:, -1] = 0
        row1 = torch.cat([convert_single(images), convert_multiple(apc)], dim=-1)
        row2 = torch.cat([convert_single(recon), convert_multiple(shp, color=boarder_color)], dim=-1)
        overview = torch.cat([row1, row2], dim=-2)
        overview = nn_func.pad(overview, [boarder_size * 4] * 4, value=1)
        overview = (overview.clamp_(0, 1) * 255).to(dtype=torch.uint8)
        return overview


def get_model(config):
    net = Model(config).cuda()
    return nn.DataParallel(net)
