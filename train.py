from random import random
from torch.utils.tensorboard import SummaryWriter
from model import save_model
from evaluation import evaluate
from tqdm import tqdm

import numpy as np
import torch
import math


class Recorder:
    def __init__(self, args, init_step, **kwargs):
        self.loss_sum = 0
        self.batch_count = 0
        self.step_count = init_step
        self.train_config = args.config['train']
        self.max_grad = args.config['train']['gradient_clipping']
        self.batch_factor = self.train_config['acml_batch_size'] / \
            self.train_config['batch_size']
        self.logger = load_logger(args)
        self.pbar = tqdm(
            initial=init_step, total=self.train_config['total_steps'], dynamic_ncols=True)
        self.loss_record = []
        self.grad_record = []
        assert self.train_config['eval_step'] >= self.train_config['log_step']

    def accumulate(self, batch_count, loss):
        self.batch_count += batch_count
        self.loss_sum += loss

    def log(self):
        self.loss_avg = np.mean(self.loss_record)
        self.grad_avg = np.mean(self.grad_record)
        self.logger.add_scalar('train_loss', self.loss_avg, self.step_count)
        self.logger.add_scalar('grad_norm', self.grad_avg, self.step_count)
        self.loss_record = []
        self.grad_record = []

    def eval(self, metrics):
        scores = ''.join(
            [' | dev_{:} {:.5f}'.format(m, s) for (m, s) in metrics])
        self.pbar.set_description(
            'train_loss {:.5f}{:}'.format(self.loss_avg, scores))
        for (m, s) in metrics:
            self.logger.add_scalar(f'dev_{m}', s, self.step_count)

    @ torch.no_grad()
    def avg_grad_norm(self, model):
        total_sqr = sum([p.grad.data.pow(2).sum().item()
                         for p in model.parameters() if p.requires_grad])
        param_nums = sum(p.numel()
                         for p in model.parameters() if p.requires_grad)
        avg_grad_norm = (total_sqr / param_nums) ** 0.5
        return avg_grad_norm

    def update(self, model):
        self.loss_record.append(self.loss_sum)
        self.grad_record.append(self.avg_grad_norm(model))
        self.clear()

    def clear(self):
        self.loss_sum = 0
        self.batch_count = 0
        self.step_count += 1
        self.pbar.update(1)

    def close(self):
        self.logger.close()
        self.pbar.close()

    def is_update(self, batch_count):
        return self.batch_count == self.train_config['acml_batch_size'] or \
            batch_count != self.train_config['batch_size']

    def is_log(self):
        return self.step_count % self.train_config['log_step'] == 0

    def is_eval(self):
        return self.step_count % self.train_config['eval_step'] == 0

    def is_stop(self):
        return self.step_count >= self.train_config['total_steps']


def load_logger(args):
    import os
    import shutil

    def process_filepath(path):
        if not args.method in path:
            if args.target_type == args.method:
                path += '/{:}'.format(args.target_type)
            else:
                path += '/{:}/{:}'.format(args.target_type, args.method)
        return path

    # build logger directory
    args.logdir = process_filepath(args.logdir)
    if os.path.isdir(args.logdir):
        shutil.rmtree(args.logdir)
    os.makedirs(args.logdir)
    logger = SummaryWriter(args.logdir)

    # build ckpt directory
    args.ckptdir = process_filepath(args.ckptdir)
    os.makedirs(args.ckptdir, exist_ok=True)
    return logger


def set_GPU_device(args, model, optim):
    model = model.to(args.device)
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(args.device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(args.device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(args.device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(
                            args.device)
    return model, optim


def train(args, model, optimizer, train_loader, dev_loader, loss_func, init_step=0):
    # set GPU device
    model, optimizer = set_GPU_device(args, model, optimizer)
    
    # build recorder
    recorder = Recorder(args, init_step)

    print('[Training] - Start training {:} model'.format(args.method))

    while not recorder.is_stop():
        # scheduler process
        for (data, targets, lengths) in train_loader:
            try:
                # load data
                data, targets = data.to(args.device), targets.to(args.device)

                # process forward and backward
                loss = model(data, targets, lengths, loss_func) / \
                    recorder.batch_factor
                loss.backward()
                loss = loss.item()

                # recording step and loss
                recorder.accumulate(len(targets), loss)
                if recorder.is_update(len(targets)):

                    # gradient clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        list(model.parameters()), recorder.max_grad)
                    if math.isnan(grad_norm) or math.isinf(grad_norm):
                        print(
                            f'[Training] - Error : grad norm is nan/inf at step {recorder.step_count}')
                        optimizer.zero_grad()
                        recorder.clear()
                        continue

                    # update loss and gradient norm recording
                    recorder.update(model)

                    # update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    if args.empty_cache:
                        torch.cuda.empty_cache()

                    # logging loss and gradient norm recording
                    if recorder.is_log():
                        recorder.log()

                    # evaluate performance on devlopment set and save model
                    if recorder.is_eval():
                        print('[Training] - Evaluating on development set')
                        results = evaluate(
                            args, dev_loader, model, loss_func, args.metric)
                        recorder.eval(results)
                        save_model(model, optimizer, args, recorder.step_count)

                    if recorder.is_stop():
                        break

            except RuntimeError as e:
                print(e)
                if not 'CUDA out of memory' in str(e):
                    raise
                print('[Training] - CUDA out of memory at step: ',
                      recorder.step_count)
                optimizer.zero_grad()
                torch.cuda.empty_cache()
    recorder.close()
