import torch
import torch.nn as nn


def init_conv(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 1e-2)


class LayerNormConv(nn.Module):

    def __init__(self, num_planes, plane_size):
        super(LayerNormConv, self).__init__()
        self.layer_norm = nn.LayerNorm([num_planes, plane_size, plane_size], elementwise_affine=False)
        self.scale = nn.Parameter(torch.ones(1, num_planes, 1, 1))
        self.offset = nn.Parameter(torch.zeros(1, num_planes, 1, 1))

    def forward(self, x):
        x = self.layer_norm(x)
        x = x * self.scale + self.offset
        return x


class EncoderConvLayer(nn.Module):

    def __init__(self, num_planes_in, num_planes_out, plane_size_out):
        super(EncoderConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(num_planes_in, num_planes_out, kernel_size=4, stride=2, padding=1),
            LayerNormConv(num_planes_out, plane_size_out),
            nn.ELU(inplace=True),
        )
        init_conv(self.modules())

    def forward(self, x):
        return self.conv(x)


class DecoderConvLayer(nn.Module):

    def __init__(self, num_planes_in, num_planes_out, plane_size_out, is_last=False):
        super(DecoderConvLayer, self).__init__()
        layers = [
            torch.nn.ZeroPad2d((0, 1, 0, 1)),
            nn.Conv2d(num_planes_in, num_planes_out, kernel_size=4, padding=1),
        ]
        if not is_last:
            layers += [
                LayerNormConv(num_planes_out, plane_size_out),
                nn.ReLU(inplace=True),
            ]
        self.conv = nn.Sequential(*layers)
        init_conv(self.modules())

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2)
        return self.conv(x)


class EncoderFCLayer(nn.Module):

    def __init__(self, num_features_in, num_features_out):
        super(EncoderFCLayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_features_in, num_features_out),
            nn.LayerNorm(num_features_out),
            nn.ELU(inplace=True),
        )

    def forward(self, x):
        return self.fc(x)


class DecoderFCLayer(nn.Module):

    def __init__(self, num_features_in, num_features_out):
        super(DecoderFCLayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_features_in, num_features_out),
            nn.LayerNorm(num_features_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.fc(x)


class Encoder(nn.Module):

    def __init__(self, config_conv, num_planes_in, plane_size_in, updater_size, normalize_inputs):
        super(Encoder, self).__init__()
        self.layer_norm = None
        if normalize_inputs:
            self.layer_norm = nn.LayerNorm([num_planes_in, plane_size_in, plane_size_in], elementwise_affine=False)
        layers = []
        for num_planes_out in config_conv:
            plane_size_out = plane_size_in // 2
            layers.append(EncoderConvLayer(num_planes_in, num_planes_out, plane_size_out))
            num_planes_in = num_planes_out
            plane_size_in = plane_size_out
        num_features_conv = num_planes_in * plane_size_in * plane_size_in
        self.conv = nn.Sequential(*layers)
        self.fc = EncoderFCLayer(num_features_conv, updater_size)

    def forward(self, x):
        if self.layer_norm:
            x = self.layer_norm(x)
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


class Decoder(nn.Module):

    def __init__(self, config_conv, num_planes_out, plane_size_out, updater_size, state_size):
        super(Decoder, self).__init__()
        layers_rev = []
        for idx, num_planes_in in enumerate(config_conv):
            layers_rev.append(DecoderConvLayer(num_planes_in, num_planes_out, plane_size_out, is_last=(idx == 0)))
            num_planes_out = num_planes_in
            plane_size_out //= 2
        num_features_conv = num_planes_out * plane_size_out * plane_size_out
        self.num_planes = num_planes_out
        self.plane_size = plane_size_out
        self.fc = nn.Sequential(
            DecoderFCLayer(state_size, updater_size),
            DecoderFCLayer(updater_size, num_features_conv),
        )
        self.conv = nn.Sequential(*reversed(layers_rev))

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.shape[0], self.num_planes, self.plane_size, self.plane_size)
        x = self.conv(x)
        return x


class UpdaterRNN(nn.Module):

    def __init__(self, input_size, state_size):
        super(UpdaterRNN, self).__init__()
        self.register_buffer('tensor', torch.FloatTensor())
        self.state_size = state_size
        self.fc = nn.Linear(input_size + state_size, state_size)
        self.layer_norm = nn.LayerNorm(state_size)

    def compute_outputs(self, states_c):
        states = torch.sigmoid(states_c)
        features = torch.sigmoid(self.layer_norm(states_c))
        return features, states

    def init_states(self, batch_size):
        states_c = self.tensor.new_empty(batch_size, self.state_size).normal_(std=1e-2)
        return self.compute_outputs(states_c)

    def forward(self, inputs, states):
        states_c = self.fc(torch.cat([inputs, states], dim=-1))
        return self.compute_outputs(states_c)


class UpdaterLSTM(nn.Module):

    def __init__(self, input_size, state_size):
        super(UpdaterLSTM, self).__init__()
        self.register_buffer('tensor', torch.FloatTensor())
        self.state_size = state_size
        self.lstm_cell = nn.LSTMCell(input_size, state_size)

    @staticmethod
    def compute_outputs(states):
        return states[0], states

    def init_states(self, batch_size):
        states_h = self.tensor.new_empty(batch_size, self.state_size).normal_(std=1e-2)
        states_c = self.tensor.new_empty(batch_size, self.state_size).normal_(std=1e-2)
        states = (states_h, states_c)
        return self.compute_outputs(states)

    def forward(self, inputs, states):
        states = self.lstm_cell(inputs, states)
        return self.compute_outputs(states)
