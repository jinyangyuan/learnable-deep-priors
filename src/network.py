import torch
import torch.nn as nn
from building_block import FC, EncoderConv, DecoderConv


def normalize_image(x):
    return x * 2 - 1


def restore_image(x):
    return (x + 1) * 0.5


class InitializerBase(nn.Module):

    def __init__(self, channel_list, kernel_list, stride_list, hidden_list, num_planes_in, plane_height_in,
                 plane_width_in, num_features_out, state_size):
        super(InitializerBase, self).__init__()
        self.net = EncoderConv(
            channel_list=channel_list,
            kernel_list=kernel_list,
            stride_list=stride_list,
            hidden_list=hidden_list,
            num_planes_in=num_planes_in,
            plane_height_in=plane_height_in,
            plane_width_in=plane_width_in,
            num_features_out=num_features_out,
            last_activation=True,
        )
        self.lstm = nn.LSTMCell(num_features_out, state_size)

    def forward(self, inputs, states=None):
        x = normalize_image(inputs)
        x = self.net(x)
        states = self.lstm(x, states)
        return states


class InitializerBack(InitializerBase):

    def __init__(self, args):
        super(InitializerBack, self).__init__(
            channel_list=args.init_back_channel_list,
            kernel_list=args.init_back_kernel_list,
            stride_list=args.init_back_stride_list,
            hidden_list=args.init_back_hidden_list,
            num_planes_in=args.image_planes,
            plane_height_in=args.image_height,
            plane_width_in=args.image_width,
            num_features_out=args.init_back_size,
            state_size=args.state_back_size,
        )


class InitializerObj(nn.Module):

    def __init__(self, args):
        super(InitializerObj, self).__init__()
        self.upd_main = InitializerBase(
            channel_list=args.init_obj_channel_list,
            kernel_list=args.init_obj_kernel_list,
            stride_list=args.init_obj_stride_list,
            hidden_list=args.init_obj_main_hidden_list,
            num_planes_in=args.image_planes * 2 + 1,
            plane_height_in=args.image_height,
            plane_width_in=args.image_width,
            num_features_out=args.init_obj_main_size,
            state_size=args.state_main_size,
        )
        self.net_obj = FC(
            hidden_list=args.init_obj_obj_hidden_list,
            num_features_in=args.state_main_size,
            num_features_out=args.init_obj_obj_size,
            activation=nn.ELU,
            last_activation=True,
        )
        self.lstm_obj = nn.LSTMCell(args.init_obj_obj_size, args.state_obj_size)

    def forward(self, inputs, states_main):
        states_main = self.upd_main(inputs, states_main)
        x = self.net_obj(states_main[0])
        states_obj = self.lstm_obj(x)
        return states_obj, states_main


class UpdaterBack(nn.Module):

    def __init__(self, args):
        super(UpdaterBack, self).__init__()
        self.net_color = FC(
            hidden_list=args.upd_back_color_hidden_list[:-1],
            num_features_in=args.image_planes,
            num_features_out=args.upd_back_color_hidden_list[-1],
            activation=nn.ELU,
            last_activation=True,
        )
        self.net_diff = EncoderConv(
            channel_list=args.upd_back_diff_channel_list,
            kernel_list=args.upd_back_diff_kernel_list,
            stride_list=args.upd_back_diff_stride_list,
            hidden_list=args.upd_back_diff_hidden_list[:-1],
            num_planes_in=args.image_planes,
            plane_height_in=args.image_height,
            plane_width_in=args.image_width,
            num_features_out=args.upd_back_diff_hidden_list[-1],
            last_activation=True,
        )
        upd_size = args.upd_back_color_hidden_list[-1] + args.upd_back_diff_hidden_list[-1]
        self.lstm = nn.LSTMCell(upd_size, args.state_back_size)

    def forward(self, inputs, states):
        x_color = self.net_color(inputs.view(*inputs.shape[:-2], -1).mean(-1))
        x_diff = self.net_diff(inputs)
        x = torch.cat([x_color, x_diff], dim=-1)
        states = self.lstm(x, states)
        return states


class UpdaterObj(nn.Module):

    def __init__(self, args):
        super(UpdaterObj, self).__init__()
        self.net_apc_color = FC(
            hidden_list=args.upd_apc_color_hidden_list[:-1],
            num_features_in=args.image_planes,
            num_features_out=args.upd_apc_color_hidden_list[-1],
            activation=nn.ELU,
            last_activation=True,
        )
        self.net_apc_diff = EncoderConv(
            channel_list=args.upd_apc_diff_channel_list,
            kernel_list=args.upd_apc_diff_kernel_list,
            stride_list=args.upd_apc_diff_stride_list,
            hidden_list=args.upd_apc_diff_hidden_list[:-1],
            num_planes_in=args.image_planes,
            plane_height_in=args.image_height,
            plane_width_in=args.image_width,
            num_features_out=args.upd_apc_diff_hidden_list[-1],
            last_activation=True,
        )
        self.net_shp = EncoderConv(
            channel_list=args.upd_shp_channel_list,
            kernel_list=args.upd_shp_kernel_list,
            stride_list=args.upd_shp_stride_list,
            hidden_list=args.upd_shp_hidden_list[:-1],
            num_planes_in=1,
            plane_height_in=args.image_height,
            plane_width_in=args.image_width,
            num_features_out=args.upd_shp_hidden_list[-1],
            last_activation=True,
        )
        upd_size = args.upd_apc_color_hidden_list[-1] + args.upd_apc_diff_hidden_list[-1] + args.upd_shp_hidden_list[-1]
        self.lstm = nn.LSTMCell(upd_size, args.state_obj_size)

    def forward(self, inputs_apc, inputs_shp, states):
        inputs_apc = inputs_apc.view(-1, *inputs_apc.shape[2:])
        inputs_shp = inputs_shp.view(-1, *inputs_shp.shape[2:])
        x_apc_color = self.net_apc_color(inputs_apc.view(*inputs_apc.shape[:-2], -1).mean(-1))
        x_apc_diff = self.net_apc_diff(inputs_apc)
        x_shp = self.net_shp(inputs_shp)
        x = torch.cat([x_apc_color, x_apc_diff, x_shp], dim=-1)
        states = self.lstm(x, states)
        return states


class DecoderBack(nn.Module):

    def __init__(self, args):
        super(DecoderBack, self).__init__()
        self.net_color = FC(
            hidden_list=reversed(args.upd_back_color_hidden_list),
            num_features_in=args.state_back_size,
            num_features_out=args.image_planes,
            activation=nn.ReLU,
            last_activation=False,
        )
        self.net_diff = DecoderConv(
            channel_list_rev=args.upd_back_diff_channel_list,
            kernel_list_rev=args.upd_back_diff_kernel_list,
            stride_list_rev=args.upd_back_diff_stride_list,
            hidden_list_rev=args.upd_back_diff_hidden_list,
            num_features_in=args.state_back_size,
            num_planes_out=args.image_planes,
            plane_height_out=args.image_height,
            plane_width_out=args.image_width,
        )

    def forward(self, x):
        back_color = self.net_color(x)[..., None, None]
        back_diff = self.net_diff(x)
        back = restore_image(back_color + back_diff)
        return {'back': back, 'back_diff': back_diff}


class DecoderObj(nn.Module):

    def __init__(self, args):
        super(DecoderObj, self).__init__()
        self.net_apc_color = FC(
            hidden_list=reversed(args.upd_apc_color_hidden_list),
            num_features_in=args.state_obj_size,
            num_features_out=args.image_planes,
            activation=nn.ReLU,
            last_activation=False,
        )
        self.net_apc_diff = DecoderConv(
            channel_list_rev=args.upd_apc_diff_channel_list,
            kernel_list_rev=args.upd_apc_diff_kernel_list,
            stride_list_rev=args.upd_apc_diff_stride_list,
            hidden_list_rev=args.upd_apc_diff_hidden_list,
            num_features_in=args.state_obj_size,
            num_planes_out=args.image_planes,
            plane_height_out=args.image_height,
            plane_width_out=args.image_width,
        )
        self.net_shp = DecoderConv(
            channel_list_rev=args.upd_shp_channel_list,
            kernel_list_rev=args.upd_shp_kernel_list,
            stride_list_rev=args.upd_shp_stride_list,
            hidden_list_rev=args.upd_shp_hidden_list,
            num_features_in=args.state_obj_size,
            num_planes_out=1,
            plane_height_out=args.image_height,
            plane_width_out=args.image_width,
        )

    def forward(self, x, num_objects):
        apc_color = self.net_apc_color(x)[..., None, None]
        apc_diff = self.net_apc_diff(x)
        logits_shp = self.net_shp(x)
        apc_color = apc_color.view(num_objects, -1, *apc_color.shape[1:])
        apc_diff = apc_diff.view(num_objects, -1, *apc_diff.shape[1:])
        logits_shp = logits_shp.view(num_objects, -1, *logits_shp.shape[1:])
        apc = restore_image(apc_color + apc_diff)
        shp = torch.sigmoid(logits_shp)
        return {'apc': apc, 'apc_diff': apc_diff, 'shp': shp, 'logits_shp': logits_shp}
