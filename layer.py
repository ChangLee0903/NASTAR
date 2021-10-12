from torchaudio.functional import magphase
from functools import partial
import torch


class OnlinePreprocessor(torch.nn.Module):
    def __init__(self, win_len=512, hop_len=256, n_freq=257, log=True, eps=1e-10, **kwargs):
        super(OnlinePreprocessor, self).__init__()
        n_fft = (n_freq - 1) * 2
        self._win_args = {'n_fft': n_fft,
                          'hop_length': hop_len, 'win_length': win_len}
        self.register_buffer('_window', torch.hann_window(win_len))

        self._stft_args = {'center': True, 'pad_mode': 'reflect',
                           'normalized': False, 'onesided': True}
        self._istft_args = {'center': True,
                            'normalized': False, 'onesided': True}

        self._stft = partial(torch.stft, **self._win_args, **self._stft_args)
        self._istft = partial(
            torch.istft, **self._win_args, **self._istft_args)
        self._magphase = partial(magphase, power=2)

        self.log = log
        self.eps = eps

    def forward(self, wavs):
        complx = self._stft(wavs, window=self._window)
        feats, phases = self._magphase(complx)
        feats = feats.sqrt()

        if self.log:
            feats = (feats + self.eps).log()

        feats = feats.permute(0, 2, 1).contiguous()
        return feats, phases

    def inverse(self, feats, phases, length=None):
        feats = feats.permute(0, 2, 1).contiguous()

        if self.log:
            feats = feats.exp()

        complxs = feats.unsqueeze(-1) * \
            torch.stack([phases.cos(), phases.sin()], dim=-1)
        return self._istft(complxs, window=self._window, length=length)


class Conv1dBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation='Identity'):
        super(Conv1dBlock, self).__init__()
        self.conv = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=kernel_size, padding=kernel_size//2)
        self.act = eval(f'torch.nn.{activation}()')

    def forward(self, features):
        predicted = self.conv(features)
        predicted = self.act(predicted)
        return predicted


class LSTM(torch.nn.Module):
    def __init__(self, feat_size, hidden_size, num_layers, bidirectional, **kwargs):
        super(LSTM, self).__init__()
        self._lstm = torch.nn.LSTM(input_size=feat_size, hidden_size=hidden_size,
                                   num_layers=num_layers, batch_first=True,
                                   bidirectional=bidirectional)
        if bidirectional:
            hidden_size = 2 * hidden_size
        self.scaling_layer = torch.nn.Linear(hidden_size, feat_size)

    def forward(self, specs):
        predicted, _ = self._lstm(specs)
        predicted = self.scaling_layer(predicted)
        predicted = torch.clamp(predicted, min=0)
        return predicted


class GRU(torch.nn.Module):
    def __init__(self, feat_size, hidden_size, num_layers, bidirectional, **kwargs):
        super(GRU, self).__init__()
        self._gru = torch.nn.GRU(input_size=feat_size, hidden_size=hidden_size,
                                 num_layers=num_layers, batch_first=True,
                                 bidirectional=bidirectional)
        if bidirectional:
            hidden_size = 2 * hidden_size
        self.scaling_layer = torch.nn.Linear(hidden_size, feat_size)

    def forward(self, specs):
        predicted, _ = self._gru(specs)
        predicted = self.scaling_layer(predicted)
        predicted = torch.clamp(predicted, min=0)
        return predicted
