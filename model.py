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


class SpectrumDenoiseModel(torch.nn.Module):
    def __init__(self, args, **kwargs):
        super(SpectrumDenoiseModel, self).__init__()
        from layer import OnlinePreprocessor
        self.preprocessor = OnlinePreprocessor(
            **args.config['model'][args.model]['feat'])
        feat_size = args.config['model'][args.model]['feat']['n_freq']
        if args.ae_ckpt is None:
            if args.model == 'LSTM':
                from layer import LSTM
                self.ae_model = LSTM(feat_size=feat_size,
                                     **args.config['model']['LSTM'])
            elif args.model == 'GRU':
                from layer import GRU
                self.ae_model = GRU(feat_size=feat_size,
                                    **args.config['model']['GRU'])

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

    @ torch.no_grad()
    def _mask(self, feat, lengths):
        stft_lengths = lengths // self.preprocessor._win_args['hop_length'] + 1
        stft_length_masks = self._get_length_masks(stft_lengths)
        feat = feat * stft_length_masks
        return feat

    def forward(self, wav, target, lengths, loss_fn):
        with torch.no_grad():
            feat_tar, _ = self.preprocessor(target)
            if not lengths is None:
                feat_tar = self._mask(feat_tar, lengths)

        predicted = self.transform(wav, lengths, True)
        loss = loss_fn(predicted.flatten(start_dim=1).contiguous(),
                       feat_tar.flatten(start_dim=1).contiguous())
        return loss

    def transform(self,  wav, lengths=None, istrain=False):
        with torch.no_grad():
            feat_inp, phase_inp = self.preprocessor(wav)
        predicted = self.ae_model(feat_inp)

        if not lengths is None:
            predicted = self._mask(predicted, lengths)
        return predicted if istrain \
            else self.preprocessor.inverse(predicted, phase_inp, wav.shape[-1])


class DenoiseModel(torch.nn.Module):
    def __init__(self, args, **kwargs):
        super(DenoiseModel, self).__init__()
        if args.ae_ckpt is None:
            if args.model == 'DEMUCS':
                from denoiser.demucs import Demucs as DEMUCS
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
        super(NoiseClassifier, self).__init__()
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
        super(DATModel, self).__init__(args, **kwargs)
        from data import NoiseTypeDataset
        dataset = NoiseTypeDataset(args)
        self.target_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.config['train']['batch_size'],
            shuffle=True,
            collate_fn=dataset.collate_fn,
            num_workers=args.n_jobs)
        self.noise_cls = NoiseClassifier()

    def forward(self, wav, target, lengths, loss_fn):
        predicted = self.transform(wav, lengths)
        se_loss = loss_fn(predicted, target)
        with torch.no_grad():
            noisy, label = next(iter(self.target_loader))
            noisy, label = noisy.to(wav.device), label.to(wav.device)
        _, hidden = self.transform(noisy, lengths, True)
        adv_loss = self.noise_cls(hidden, label)
        loss = se_loss + adv_loss
        return loss

    def transform(self, wav, lengths=None, is_train=False):
        predicted, hidden = self.ae_model(wav.unsqueeze(1))
        predicted = predicted.squeeze(1)
        if is_train:
            return predicted, hidden
        if not lengths is None:
            length_masks = self._get_length_masks(lengths).to(wav.device)
            predicted = predicted * length_masks
        return predicted
