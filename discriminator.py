from pesq import pesq, PesqError
from joblib import Parallel, delayed
from evaluation import stoi_eval, pesq_wb_eval, pesq_nb_eval
from torch import nn
from torch.nn.utils import spectral_norm
from functools import partial
from packaging import version
import numpy as np
import torch
import random


def xavier_init_layer(
    in_size, out_size=None, spec_norm=True, layer_type=nn.Linear, **kwargs
):
    "Create a layer with spectral norm, xavier uniform init and zero bias"
    if out_size is None:
        out_size = in_size

    layer = layer_type(in_size, out_size, **kwargs)
    if spec_norm:
        layer = spectral_norm(layer)

    # Perform initialization
    nn.init.xavier_uniform_(layer.weight, gain=1.0)
    nn.init.zeros_(layer.bias)

    return layer


class Conv1d(torch.nn.Conv1d):
    """Conv1d module with customized initialization."""

    def __init__(self, *args, **kwargs):
        """Initialize Conv1d module."""
        super(Conv1d, self).__init__(*args, **kwargs)

    def reset_parameters(self):
        """Reset parameters."""
        torch.nn.init.kaiming_normal_(self.weight, nonlinearity="relu")
        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0.0)


class ParallelWaveGANDiscriminator(torch.nn.Module):
    """Parallel WaveGAN Discriminator module."""

    def __init__(
        self,
        in_channels=2,
        out_channels=1,
        kernel_size=3,
        layers=10,
        conv_channels=64,
        dilation_factor=1,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.2},
        bias=True,
        use_weight_norm=True,
    ):
        """Initialize Parallel WaveGAN Discriminator module.
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Number of output channels.
            layers (int): Number of conv layers.
            conv_channels (int): Number of chnn layers.
            dilation_factor (int): Dilation factor. For example, if dilation_factor = 2,
                the dilation will be 2, 4, 8, ..., and so on.
            nonlinear_activation (str): Nonlinear function after each conv.
            nonlinear_activation_params (dict): Nonlinear function parameters
            bias (bool): Whether to use bias parameter in conv.
            use_weight_norm (bool) Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.
        """
        super(ParallelWaveGANDiscriminator, self).__init__()
        assert (kernel_size - 1) % 2 == 0, "Not support even number kernel size."
        assert dilation_factor > 0, "Dilation factor must be > 0."
        self.conv_layers = torch.nn.ModuleList()
        conv_in_channels = in_channels
        for i in range(layers - 1):
            if i == 0:
                dilation = 1
                stride = 1
            else:
                dilation = i if dilation_factor == 1 else dilation_factor ** i
                conv_in_channels = conv_channels
                stride = 2

            padding = (kernel_size - 1) // 2 * dilation
            conv_layer = [
                Conv1d(
                    conv_in_channels,
                    conv_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    dilation=dilation,
                    stride=stride,
                    bias=bias,
                ),
                getattr(torch.nn, nonlinear_activation)(
                    inplace=True, **nonlinear_activation_params
                )
            ]
            self.conv_layers += conv_layer
        padding = (kernel_size - 1) // 2

        last_conv_layer = Conv1d(
            conv_in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=bias,
        )
        self.conv_layers += [last_conv_layer]

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

    def forward(self, x):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input noise signal (B, 1, T).
        Returns:
            Tensor: Output tensor (B, 1, T)
        """
        for f in self.conv_layers:
            x = f(x)
        x = x.mean(dim=(1, 2))
        return x

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)


class STFT(torch.nn.Module):
    """computes the Short-Term Fourier Transform (STFT).
    This class computes the Short-Term Fourier Transform of an audio signal.
    It supports multi-channel audio inputs (batch, time, channels).
    Arguments
    ---------
    sample_rate : int
        Sample rate of the input audio signal (e.g 16000).
    win_length : float
        Length (in ms) of the sliding window used to compute the STFT.
    hop_length : float
        Length (in ms) of the hope of the sliding window used to compute
        the STFT.
    n_fft : int
        Number of fft point of the STFT. It defines the frequency resolution
        (n_fft should be <= than win_len).
    window_fn : function
        A function that takes an integer (number of samples) and outputs a
        tensor to be multiplied with each window before fft.
    normalized_stft : bool
        If True, the function returns the  normalized STFT results,
        i.e., multiplied by win_length^-0.5 (default is False).
    center : bool
        If True (default), the input will be padded on both sides so that the
        t-th frame is centered at time t×hop_length. Otherwise, the t-th frame
        begins at time t×hop_length.
    pad_mode : str
        It can be 'constant','reflect','replicate', 'circular', 'reflect'
        (default). 'constant' pads the input tensor boundaries with a
        constant value. 'reflect' pads the input tensor using the reflection
        of the input boundary. 'replicate' pads the input tensor using
        replication of the input boundary. 'circular' pads using  circular
        replication.
    onesided : True
        If True (default) only returns nfft/2 values. Note that the other
        samples are redundant due to the Fourier transform conjugate symmetry.
    Example
    -------
    >>> import torch
    >>> compute_STFT = STFT(
    ...     sample_rate=16000, win_length=25, hop_length=10, n_fft=400
    ... )
    >>> inputs = torch.randn([10, 16000])
    >>> features = compute_STFT(inputs)
    >>> features.shape
    torch.Size([10, 101, 201, 2])
    """

    def __init__(
        self,
        sample_rate=16000,
        win_length=32,
        hop_length=16,
        n_fft=512,
        window_fn=torch.hamming_window,
        normalized_stft=False,
        center=True,
        pad_mode="constant",
        onesided=True,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.normalized_stft = normalized_stft
        self.center = center
        self.pad_mode = pad_mode
        self.onesided = onesided

        # Convert win_length and hop_length from ms to samples
        self.win_length = int(
            round((self.sample_rate / 1000.0) * self.win_length)
        )
        self.hop_length = int(
            round((self.sample_rate / 1000.0) * self.hop_length)
        )

        self.window = window_fn(self.win_length)

    def forward(self, x):
        """Returns the STFT generated from the input waveforms.
        Arguments
        ---------
        x : tensor
            A batch of audio signals to transform.
        """

        # Managing multi-channel stft
        or_shape = x.shape
        if len(or_shape) == 3:
            x = x.transpose(1, 2)
            x = x.reshape(or_shape[0] * or_shape[2], or_shape[1])

        if version.parse(torch.__version__) <= version.parse("1.6.0"):
            stft = torch.stft(
                x,
                self.n_fft,
                self.hop_length,
                self.win_length,
                self.window.to(x.device),
                self.center,
                self.pad_mode,
                self.normalized_stft,
                self.onesided,
            )
        else:
            stft = torch.stft(
                x,
                self.n_fft,
                self.hop_length,
                self.win_length,
                self.window.to(x.device),
                self.center,
                self.pad_mode,
                self.normalized_stft,
                self.onesided,
                return_complex=False,
            )

        # Retrieving the original dimensionality (batch,time, channels)
        if len(or_shape) == 3:
            stft = stft.reshape(
                or_shape[0],
                or_shape[2],
                stft.shape[1],
                stft.shape[2],
                stft.shape[3],
            )
            stft = stft.permute(0, 3, 2, 4, 1)
        else:
            # (batch, time, channels)
            stft = stft.transpose(2, 1)

        return stft


def spectral_magnitude(stft, power=1, log=False, eps=1e-14):
    """Returns the magnitude of a complex spectrogram.
    Arguments
    ---------
    stft : torch.Tensor
        A tensor, output from the stft function.
    power : int
        What power to use in computing the magnitude.
        Use power=1 for the power spectrogram.
        Use power=0.5 for the magnitude spectrogram.
    log : bool
        Whether to apply log to the spectral features.
    Example
    -------
    >>> a = torch.Tensor([[3, 4]])
    >>> spectral_magnitude(a, power=0.5)
    tensor([5.])
    """
    spectr = stft.pow(2).sum(-1)

    # Add eps avoids NaN when spectr is zero
    if power < 1:
        spectr = spectr + eps
    spectr = spectr.pow(power)

    if log:
        return torch.log(spectr + eps)
    return spectr


class MetricDiscriminator(nn.Module):
    """Metric estimator for enhancement training.
    Consists of:
     * four 2d conv layers
     * channel averaging
     * three linear layers
    Arguments
    ---------
    kernel_size : tuple
        The dimensions of the 2-d kernel used for convolution.
    base_channels : int
        Number of channels used in each conv layer.
    """

    def __init__(
        self, kernel_size=(5, 5), base_channels=15, activation=nn.LeakyReLU,
    ):
        super().__init__()

        self.activation = activation(negative_slope=0.3)

        self.BN = nn.BatchNorm2d(num_features=2, momentum=0.01)

        self.conv1 = xavier_init_layer(
            2, base_channels, layer_type=nn.Conv2d, kernel_size=kernel_size
        )
        self.conv2 = xavier_init_layer(
            base_channels, layer_type=nn.Conv2d, kernel_size=kernel_size
        )
        self.conv3 = xavier_init_layer(
            base_channels, layer_type=nn.Conv2d, kernel_size=kernel_size
        )
        self.conv4 = xavier_init_layer(
            base_channels, layer_type=nn.Conv2d, kernel_size=kernel_size
        )

        self.Linear1 = xavier_init_layer(base_channels, out_size=50)
        self.Linear2 = xavier_init_layer(in_size=50, out_size=10)
        self.Linear3 = xavier_init_layer(in_size=10, out_size=1)

        self._stft = STFT()

    def compute_feat(self, x):
        x = self._stft(x)
        x = spectral_magnitude(x, power=0.5)
        x = torch.log1p(x)
        return x

    def forward(self, x):
        x = torch.stack([self.compute_feat(x[:, 0, :]),
                        self.compute_feat(x[:, 1, :])], dim=1)
        out = self.BN(x)

        out = self.conv1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.activation(out)

        out = self.conv4(out)
        out = self.activation(out)
        out = torch.mean(out, (2, 3))

        out = self.Linear1(out)
        out = self.activation(out)

        out = self.Linear2(out)
        out = self.activation(out)

        out = self.Linear3(out)

        return out.view(-1)


def pesq_wb(src, tar, sr=16000):
    src, tar = src.numpy(), tar.numpy()
    assert src.ndim == 1 and tar.ndim == 1
    try:
        mos_lqo = pesq(sr, tar, src / np.abs(src).max(), 'wb')
    except PesqError as e:
        mos_lqo = 0
    return mos_lqo

from loss import L1_MRSTFTLoss


class MetricEstimator(torch.nn.Module):
    def __init__(self, args, **kwargs):
        super(MetricEstimator, self).__init__()
        self.discriminator = ParallelWaveGANDiscriminator()
        self.replay_buffer = None

        self.replay_buffer = []
        self.queue_size = 16384

        self.loss = args.loss
        self.min_score = -1.0
        self.mrl1 = L1_MRSTFTLoss(**args.config['model']['DEMUCS']['stftloss'])
        random.seed(args.seed)

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'weight' in name:
                if param.data.ndim == 1:
                    param.data = param.data.unsqueeze(-1)
                    torch.nn.init.xavier_uniform_(param.data)
                    param.data = param.data.squeeze(-1)
                else:
                    torch.nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                torch.nn.init.constant_(param.data, 0)

    @ torch.no_grad()
    def get_score(self, predicted, target):
        if predicted.shape[0] == 1:
            real_score = [pesq_wb_eval(
                p, t) for p, t in zip(predicted.cpu(), target.cpu())]
        else:
            real_score = Parallel(n_jobs=len(predicted))(delayed(pesq_wb_eval)(p, t)
                                                         for p, t in zip(predicted.cpu(), target.cpu()))
        real_score = -(torch.Tensor(real_score) + 0.5) / 5
        return real_score

    @ torch.no_grad()
    def update_replay_queue(self, predicted, target, real_score):
        batch = [(p.detach().cpu(), t.detach().cpu(), s)
                 for p, t, s in zip(predicted, target, real_score)]
        batch_size = len(batch)
        if self.queue_size % batch_size == 0:
            if len(self.replay_buffer) < self.queue_size:
                self.replay_buffer += batch
            else:
                self.replay_buffer = \
                    self.replay_buffer[batch_size:] + batch

            random.shuffle(self.replay_buffer)

    def sample_replay(self, sample_num):
        rply_predicted = []
        rply_target = []
        rply_score = []
        for (p, t, s) in random.sample(self.replay_buffer, sample_num):
            rply_predicted.append(p)
            rply_target.append(t)
            rply_score.append(s)

        rply_predicted = torch.stack(rply_predicted)
        rply_target = torch.stack(rply_target)
        rply_score = torch.Tensor(rply_score)
        return rply_predicted, rply_target, rply_score

    def forward(self, input, target, model_G=None):
        if model_G is None:
            predicted = self.transform(input, target)
            predicted = torch.clamp(predicted, min=-1.0, max=0.0)
            loss = 0.3 * predicted.mean() + self.mrl1(input, target)
            return loss
        else:
            with torch.no_grad():
                predicted = model_G.transform(input)

        niy_real_score = self.get_score(input, target).to(predicted.device)
        niy_est_score = self.transform(input, target)
        niy_diff_score = niy_real_score - niy_est_score
        niy_loss = niy_diff_score.abs().mean()

        real_score = self.get_score(predicted, target).to(predicted.device)
        neg_est_score = self.transform(predicted, target)
        neg_diff_score = real_score - neg_est_score
        neg_loss = neg_diff_score.abs().mean()

        # alpha = 0.5
        # neg_loss_p = alpha * \
        #     torch.clamp(neg_diff_score, max=None, min=0).mean() * 2
        # neg_loss_n = - (1 - alpha) * \
        #     torch.clamp(neg_diff_score, min=None, max=0).mean() * 2
        # neg_loss = neg_loss_p + neg_loss_n

        pos_est_score = self.transform(target, target)
        pos_diff_score = pos_est_score - self.min_score
        # pos_diff_score = torch.clamp(pos_diff_score, max=None, min=0)
        pos_loss = pos_diff_score.abs().mean()

        print('ref:', -real_score * 5 - 0.5)
        print('neg:', -neg_est_score * 5 - 0.5)
        print('pos:', -pos_est_score * 5 - 0.5)
        print('---------------------------------------')

        loss = neg_loss + pos_loss + niy_loss

        if not self.replay_buffer is None:
            if len(self.replay_buffer) > len(predicted):
                rply_predicted, rply_target, rply_real_score = self.sample_replay(
                    len(predicted))
                rply_predicted, rply_target, rply_real_score = rply_predicted.to(
                    predicted.device), rply_target.to(predicted.device), rply_real_score.to(predicted.device)
                rply_est_score = self.transform(rply_predicted, rply_target)
                rply_diff_score = rply_est_score - rply_real_score
                rply_loss = rply_diff_score.abs().mean()
                # rply_loss = torch.clamp(rply_diff_score, max=None, min=0).mean()
                loss = loss + rply_loss
            self.update_replay_queue(predicted, target, real_score)
        return loss

    def transform(self, predicted, target):
        input = torch.stack([predicted, target], dim=1)
        est_score = self.discriminator(input)
        return est_score
