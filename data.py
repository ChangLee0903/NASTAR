from librosa.util import find_files
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import torch
import os
import glob
import random
import librosa


def readfile(path, sr=16000):
    if '.npy' in path:
        return torch.FloatTensor(np.load(path))
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
            if random.uniform(0, 1) > self.alpha:
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
        if mode == 'dev':
            self.signal_list = random.choices(
                self.signal_list, k=args.valid_num)
        mode = 'train' if self.istrain else 'eval'

        if self.istrain:
            self.min_length = args.config['train']['min_length']
            self.max_length = args.config['train']['max_length']
        if 'DAT' in args.method or args.method == 'PTN':
            noise_list = filestrs2list(
                args.config['dataset']['train']['noise'])
        elif (not args.use_source_noise or not self.istrain):
            noise_list = None
        else:
            noise_list = filestrs2list(
                args.config['dataset'][mode]['noise'])
            if not 'ALL' in args.method and not 'DAT' in args.method:
                noise_list = [n for n in noise_list if n.split(
                    '/')[-1] in args.cohort_list]

        

        self.corruptor = Corruptor(
            noise_list, **args.config[mode]['Corruptor'],
            alpha=args.alpha, seed=args.seed, target_noise=args.target_noise
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


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        noisy_path = os.path.join(
            args.config['dataset']['test']['data'], args.target_type, 'noisy')
        clean_path = os.path.join(
            args.config['dataset']['test']['data'], args.target_type, 'clean')

        self.noisy_list = [os.path.join(noisy_path, p)
                           for p in os.listdir(noisy_path)]
        self.clean_list = [os.path.join(clean_path, p)
                           for p in os.listdir(clean_path)]
        random.seed(args.seed)

    def __len__(self):
        return len(self.clean_list)

    def __getitem__(self, idx):
        noisy = readfile(self.noisy_list[idx])
        clean = readfile(self.clean_list[idx])
        return noisy, clean

    def collate_fn(self, data):
        noisy = pad_sequence(
            [wav[0] for wav in data], batch_first=True).contiguous()
        clean = pad_sequence(
            [wav[1] for wav in data], batch_first=True).contiguous()
        lengths = torch.LongTensor([len(wav[0]) for wav in data])
        return noisy, clean, lengths


class NoiseTypeDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        random.seed(args.seed)
        signal_path = os.path.join(
            args.config['dataset']['test']['data'], args.target_type, 'noisy')
        if 'full' in args.method:
            self.signal_list = [os.path.join(signal_path, p)
                                    for p in os.listdir(signal_path)]
        elif 'one' in args.method:
            self.signal_list = [os.path.join(signal_path,
                                                os.listdir(signal_path)[0])] * len(os.listdir(signal_path))

        self.min_length = args.config['train']['min_length']
        self.max_length = args.config['train']['max_length']

    def __len__(self):
        return len(self.signal_list)

    def __getitem__(self, idx):
        noisy = self.truncate(readfile(self.signal_list[idx]))
        return noisy

    def truncate(self, sig):
        seg_length = random.randint(self.min_length, self.max_length)
        pos = random.randrange(max(1, len(sig) - seg_length))
        sig = sig[pos: pos+seg_length]
        return sig

    def collate_fn(self, data):
        noisy = pad_sequence(
            [d for d in data], batch_first=True).contiguous()
        return noisy


def get_dataloader(args, mode='train'):
    is_train = mode == 'train'
    if args.task == 'train' or args.task == 'dev':
        dataset = DenoisingDataset(args, mode)
    elif args.task == 'test' or args.task == 'write':
        dataset = TestDataset(args)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.config['train']['batch_size'],
        shuffle=is_train,
        collate_fn=dataset.collate_fn,
        num_workers=args.n_jobs)

    return dataloader
