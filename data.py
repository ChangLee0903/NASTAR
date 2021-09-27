from librosa.util import find_files
from torch.functional import split
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import glob
import random
import librosa


def readfile(path, sr=16000):
    if '.npy' in path:
        return torch.FloatTensor(np.load(path)[:, 0])
    elif '.wav' in path or '.flac' in path:
        wav, sr = librosa.load(path, sr=sr)
        wav = torch.FloatTensor(wav)
        wav = wav / wav.abs().max()
        return wav


class Corruptor:
    def __init__(self, noise_list, snrs=None, duplicate=False, seed=5566, target_noise=None, alpha=0.5, **kwargs):
        if not noise_list is None:
            assert not snrs is None

        self.noise_list = noise_list
        self.target_noise = target_noise
        self.alpha = alpha
        self.snrs = snrs
        self.isduplicate = duplicate
        random.seed(seed)

    def duplicate(self, wav, length):
        if length >= wav.size(-1):
            times = length // wav.size(-1) + 1
            wav = wav.repeat(times)
        return wav

    def noise_scaling(self, speech, noise, eps=1e-10):
        snr = random.choice(self.snrs)
        snr_exp = 10.0 ** (snr / 10.0)
        speech_power = speech.pow(2).sum(dim=-1, keepdim=True)
        noise_power = noise.pow(2).sum(dim=-1, keepdim=True)
        scalar = (speech_power / (snr_exp * noise_power + eps)).pow(0.5)
        scaled_noise = scalar * noise
        return scaled_noise

    def sample_noise(self):
        if not self.target_noise is None and not self.noise_list is None:
            if random.choices([False, True],
                              weights=[self.alpha, 1-self.alpha], k=1):
                noise = self.target_noise
            else:
                noise = readfile(random.choice(self.noise_list))
        elif not self.target_noise is None:
            noise = self.target_noise
        else:
            noise = readfile(random.choice(self.noise_list))
        return noise

    def corrupt(self, speech):
        add_noise = self.sample_noise()
        if self.isduplicate:
            add_noise = self.duplicate(add_noise, speech.size(-1))
        scaled_noise = self.noise_scaling(speech, add_noise)

        pos = random.randint(
            0, max(0, abs(speech.size(-1)-add_noise.size(-1))-1))
        if add_noise.size(-1) >= speech.size(-1):
            noisy = speech + scaled_noise[pos:pos+speech.size(-1)]
        else:
            noisy = speech[:]
            noisy[pos:pos + add_noise.size(-1)] += scaled_noise

        return noisy, scaled_noise


def filestrs2list(filestrs, fileroot=None, sample_num=0, select_sampled=False, querys=None, base_path=None, **kwargs):
    path = filestrs
    if type(filestrs) is not list:
        filestrs = [filestrs]

    all_files = []
    for filestr in filestrs:
        if os.path.isdir(filestr):
            all_files += sorted(find_files(filestr))
        elif os.path.isfile(filestr):
            with open(filestr, 'r') as handle:
                all_files += sorted(
                    [f'{fileroot}/{line[:-1]}' for line in handle.readlines()])
        else:
            all_files += sorted(glob.glob(filestr))

    all_files = sorted(all_files)
    print(
        f'[Filestrs2List] - Parsing filestrs: {path}. Complete parsing: {len(all_files)} files found.')
    return all_files


class DenoisingDataset(torch.utils.data.Dataset):
    def __init__(self, args, mode):
        self.istrain = mode == 'train'
        random.seed(args.seed)

        self.signal_list = filestrs2list(
            args.config['dataset'][mode]['speech'])

        if self.istrain:
            self.min_length = args.config['train']['min_length']
            self.max_length = args.config['train']['max_length']
        mode = 'train' if self.istrain else 'eval'

        if args.method in ['GT', 'EXTR'] or not self.istrain:
            noise_list = None
        elif args.method in ['RETV', 'NASTAR']:
            noise_list = filestrs2list(
                args.config['dataset'][mode]['noise'])
            noise_list = [n for n in noise_list if n.split(
                '/')[-1] in args.cohort_list]

        self.corruptor = Corruptor(
            noise_list, **args.config[mode]['Corruptor'],
            seed=args.seed, target_noise=args.target_noise
            if self.istrain else args.eval_noise)

    def __len__(self):
        return len(self.signal_list)

    def __getitem__(self, idx):
        clean = readfile(self.signal_list[idx])
        if self.istrain:
            clean = self.truncate(clean)
        noisy, scaled_noise = self.corruptor.corrupt(clean)
        return noisy, clean

    def truncate(self, sig):
        seg_length = random.randint(self.min_length, self.max_length)
        pos = random.randrange(max(1, len(sig) - seg_length))
        sig = sig[pos: pos+seg_length]
        return sig

    def collate_fn(self, data):
        noisy = pad_sequence(
            [wav[0] for wav in data], batch_first=True).contiguous()
        clean = pad_sequence(
            [wav[1] for wav in data], batch_first=True).contiguous()
        lengths = torch.LongTensor([len(wav[0]) for wav in data])
        return noisy, clean, lengths


def get_dataloader(args, mode='train'):
    is_train = mode == 'train'
    dataset = DenoisingDataset(args, mode)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.config['train']['batch_size'],
        shuffle=is_train,
        collate_fn=dataset.collate_fn,
        num_workers=args.n_jobs)

    return dataloader