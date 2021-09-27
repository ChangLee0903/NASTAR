from pesq import pesq, PesqError
from pystoi import stoi
from joblib import Parallel, delayed
from tqdm import tqdm
from loss import PMSQE
import scipy.io.wavfile
import numpy as np
import torch

OOM_RETRY_LIMIT = 10
EPS = np.finfo(float).eps
non_parallel_metrics = ['pesq_nb', 'pesq_wb', 'stoi', 'estoi']


def pmsqe_eval(src, tar):
    with torch.no_grad():
        return pmsqe_func(src, tar).item() * len(src)


def sisdr_eval(src, tar, sr=16000):
    alpha = (src * tar).sum(dim=-1, keepdim=True) / \
        ((tar * tar).sum(dim=-1, keepdim=True) + EPS)
    ay = alpha * tar
    norm = ((ay - src) * (ay - src)).sum(dim=-1, keepdim=True) + EPS
    sisdr = 10 * ((ay * ay).sum(dim=-1, keepdim=True) / norm + EPS).log10()
    return sisdr.sum().item()


def rmse_eval(src, tar):
    pred_rmse = ((src - tar).pow(2).mean(dim=-1))**0.5
    return pred_rmse.sum().item()


def snr_eval(src, tar):
    square_sum = tar.pow(2).sum(dim=-1)
    pred_mse = (src - tar).pow(2).sum(dim=-1)
    snr_out = 10 * (square_sum / pred_mse).log10()
    return snr_out.sum().item()


def prd_eval(src, tar, org=None):
    pred_mse = (src - tar).pow(2).sum(dim=-1)
    square_sum = tar.pow(2).sum(dim=-1)
    prd = (100 * (pred_mse / square_sum)**0.5).sum().item()
    return prd


def pesq_nb_eval(src, tar, sr=16000):
    src, tar = src.numpy(), tar.numpy()
    assert src.ndim == 1 and tar.ndim == 1
    try:
        mos_lqo = pesq(sr, tar, src/np.abs(src).max(), 'nb')
    except PesqError as e:
        mos_lqo = 0
    return mos_lqo


def pesq_wb_eval(src, tar, sr=16000):
    src, tar = src.numpy(), tar.numpy()
    assert src.ndim == 1 and tar.ndim == 1
    try:
        mos_lqo = pesq(sr, tar, src, 'wb')
    except PesqError as e:
        mos_lqo = 0
    return mos_lqo


def stoi_eval(src, tar, sr=16000):
    src, tar = src.numpy(), tar.numpy()
    assert src.ndim == 1 and tar.ndim == 1
    stoi_lqo = max(0.3, stoi(tar, src, sr, extended=False))
    return stoi_lqo


def estoi_eval(src, tar, sr=16000):
    src, tar = src.numpy(), tar.numpy()
    assert src.ndim == 1 and tar.ndim == 1
    stoi_lqo = max(0.3, stoi(tar, src, sr, extended=True))
    return stoi_lqo


def calculate_metric(length, predicted, targets, metric_fn):
    return metric_fn(predicted[0, :length], targets[0, :length])


def non_parallel_cal(args, data, targets, lengths, model, metrics):
    batch_size = targets.shape[0]

    ones = torch.ones(batch_size).long().unsqueeze(
        0).expand(len(metrics), -1)
    metric_ids = ones * \
        torch.arange(len(metrics)).unsqueeze(-1)
    metric_fns = [metrics[idx.item()]
                  for idx in metric_ids.reshape(-1)]
    predicted = model.transform(data)
    predicted_list = predicted.squeeze().detach().cpu().chunk(batch_size) * len(metrics)

    targets_list = targets.squeeze().detach(
    ).cpu().chunk(batch_size) * len(metrics)

    lengths_list = lengths.detach().cpu().chunk(batch_size) * len(metrics)

    scores = Parallel(n_jobs=args.n_jobs)(delayed(calculate_metric)(l, p, t, f)
                                          for l, p, t, f in zip(lengths_list, predicted_list, targets_list, metric_fns))
    scores = torch.FloatTensor(scores).view(
        len(metrics), batch_size).sum(dim=1)
    return scores


def parallel_cal(args, data, targets, lengths, model, metrics):
    predicted = model.transform(data, lengths)
    scores = torch.FloatTensor(
        [m(predicted, targets) for m in metrics])
    return scores


def evaluate(args, dataloader, model, loss_func, cal_metric=False):
    cal_metric = True
    metric_lst = args.config['eval']['metrics']
    loss_sum = 0

    if cal_metric and 'pmsqe' in metric_lst:
        global pmsqe_func
        pmsqe_func = PMSQE()
        pmsqe_func = pmsqe_func.to(args.device)

    model.to(args.device)
    model.eval()

    if cal_metric:
        np_scores, p_scores = torch.Tensor([]), torch.Tensor([])
        np_metrics = [eval(f'{m}_eval')
                      for m in metric_lst if m in non_parallel_metrics]
        p_metrics = [eval(f'{m}_eval')
                     for m in metric_lst if not m in non_parallel_metrics]
        scores_sum = torch.zeros(len(np_metrics + p_metrics))
    
    oom_counter = 0
    n_sample = 0
    with torch.no_grad():
        for (data, targets, lengths) in tqdm(dataloader, desc="Iteration"):
            try:
                # load data and compute loss
                data, targets = data.to(args.device), targets.to(args.device)
                loss_sum += model(data, targets, lengths, loss_func).item()

                if cal_metric:
                    if len(np_metrics) != 0:
                        np_scores = non_parallel_cal(args,
                                                     data, targets, lengths, model, np_metrics)
                    if len(p_metrics) != 0:
                        p_scores = parallel_cal(
                            args, data, targets, lengths, model, p_metrics)
                    scores_sum += torch.cat([np_scores, p_scores])
                
                # compute n_sample
                n_sample += len(targets)

            except RuntimeError as e:
                print(e)
                if not 'CUDA out of memory' in str(e):
                    raise
                if oom_counter >= OOM_RETRY_LIMIT:
                    oom_counter = 0
                    break
                oom_counter += 1
                torch.cuda.empty_cache()

    model.train()
    torch.cuda.empty_cache()
    
    loss_avg = loss_sum / n_sample
    metrics = [('loss', loss_avg)]
    if cal_metric:
        scores_avg = scores_sum / n_sample
        metric_keys = [m for m in metric_lst if m in non_parallel_metrics] + \
            [m for m in metric_lst
                if not m in non_parallel_metrics]
        metrics += [(m, s) for (m, s) in zip(metric_keys, scores_avg)]
    return metrics


def normalize(wav, target_level=-25):
    wav = wav / np.abs(wav).max()
    rms = (wav ** 2).mean() ** 0.5
    scalar = 10 ** (target_level / 20) / (rms+EPS)
    wav = wav * scalar
    return wav


def write_wav(dir, i, niy, tar, pre):
    niy = normalize(niy)
    tar = normalize(tar)
    pre = normalize(pre)

    scipy.io.wavfile.write(f'{dir}/sample_{i}_niy.wav', 16000, niy)
    scipy.io.wavfile.write(f'{dir}/sample_{i}_tar.wav', 16000, tar)
    scipy.io.wavfile.write(f'{dir}/sample_{i}_pre.wav', 16000, pre)


def write(args, dataloader, model):
    import os
    os.makedirs(args.out, exist_ok=True)

    model.to(args.device)
    model.eval()

    n_sample = 0
    with torch.no_grad():
        for (data, targets) in dataloader:
            data = data.to(args.device)
            predicted = model.transform(data)

            data = data.cpu().numpy()
            targets = targets.cpu().numpy()
            predicted = predicted.cpu().numpy()

            for niy, tar, pre in zip(data, targets, predicted):
                n_sample += 1
                write_wav(args.out, n_sample, niy, tar, pre)

                if not args.sample_num is None and n_sample >= args.sample_num:
                    return
