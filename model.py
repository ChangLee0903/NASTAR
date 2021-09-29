from torch.autograd import Function
import torch.nn.functional as F
import torch
import random
MAX_POSITIONS_LEN = 16000 * 50


def save_model(model, optimizer, args, current_step):
    path = f'{args.ckptdir}/SE_{args.model}_{current_step}.pth'
    all_states = {
        'Model': model,
        'Optimizer': optimizer.state_dict(),
        'Current_step': current_step,
        'Args': args
    }
    torch.save(all_states, path)


def load_model(args):
    ckpt = torch.load(args.ckpt, map_location='cpu')

    args = ckpt['Args']
    current_step = ckpt['Current_step']

    model = ckpt['Model']
    optimizer = eval(f'torch.optim.{args.opt}')(model.parameters(),
                                                **args.config['optimizer'][args.opt])
    optimizer.load_state_dict(ckpt['Optimizer'])

    return args, model, optimizer, current_step


class Conv1dBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, spectral_norm=False, activation='Identity'):
        super(Conv1dBlock, self).__init__()
        self.conv = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=kernel_size, padding=kernel_size//2)
        if spectral_norm:
            self.conv = torch.nn.utils.spectral_norm(self.conv)
        self.act = eval(f'torch.nn.{activation}()')

    def forward(self, features):
        predicted = self.conv(features)
        predicted = self.act(predicted)
        return predicted


class DenoiseModel(torch.nn.Module):
    def __init__(self, args, **kwargs):
        super(DenoiseModel, self).__init__()
        if args.ae_ckpt is None:
            if args.model == 'DEMUCS':
                from denoiser.demucs import Demucs as DEMUCS
                self.ae_model = DEMUCS(**args.config['model']['DEMUCS'])
                self.init_weights()
            elif args.model == 'LSTM':
                self.ae_model = DEMUCS(**args.config['model']['DEMUCS'])
                self.init_weights()
        else:
            self.ae_model = torch.load(args.ae_ckpt)

    @ torch.no_grad()
    def _get_length_masks(self, lengths):
        # lengths: (batch_size, ) in cuda
        ascending = torch.arange(MAX_POSITIONS_LEN)[:lengths.max().item()].unsqueeze(
            0).expand(len(lengths), -1)
        length_masks = (ascending < lengths.unsqueeze(-1))
        return length_masks

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

    def forward(self, wav, target, lengths, loss_fn):
        predicted = self.transform(wav, lengths)
        loss = loss_fn(predicted, target)
        return loss

    def transform(self, wav, lengths=None):
        predicted, _ = self.ae_model(wav.unsqueeze(1))
        predicted = predicted.squeeze(1)
        if not lengths is None:
            length_masks = self._get_length_masks(lengths).to(wav.device)
            predicted = predicted * length_masks
        return predicted


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class NoiseClassifier(torch.nn.Module):
    def __init__(self, **kwargs):
        self.lstm = torch.nn.LSTM(input_size=768, hidden_size=256,
                                  num_layers=1, batch_first=True, bidirectional=True)
        self.fcn = torch.nn.Linear(512, 5)
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, hidden, label):
        hidden = ReverseLayerF.apply(hidden, 0.05)
        hidden, _ = self.lstm(hidden)
        hidden = hidden.mean(dim=1)
        logit = self.fcn(hidden)
        loss = self.loss(logit, label)
        return loss


class DATModel(DenoiseModel):
    def __init__(self, args, **kwargs):
        super(DenoiseModel, self).__init__(args)
        from data import NoiseTypeDataset
        self.target_loader = torch.utils.data.DataLoader(
            NoiseTypeDataset(args),
            batch_size=args.config['train']['batch_size'],
            shuffle=True,
            num_workers=args.n_jobs)
        self.noise_cls = NoiseClassifier()

    def forward(self, wav, target, lengths, loss_fn):
        predicted, _ = self.transform(wav, lengths)
        se_loss = loss_fn(predicted, target)

        noisy, label = next(iter(self.target_loader))
        _, hidden = self.transform(noisy, lengths)
        adv_loss = self.noise_cls(hidden, label)

        loss = se_loss + adv_loss
        return loss
