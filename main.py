from data import get_dataloader, readfile
import numpy as np
import argparse
import yaml
import torch
import random


def set_random_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def argument_parsing():
    print('[Parsing] - Start Argument Parsing')

    parser = argparse.ArgumentParser(description='Argument Parser.')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to experiment configuration.')
    parser.add_argument('--model', type=str, default='DEMUCS',
                        choices=['DEMUCS', 'LSTM', 'GRU'], help='denoising model type.')
    parser.add_argument('--method', type=str, help='Method for noise adaptation.')
    parser.add_argument('--task', type=str, default='train',
                        choices=['train', 'test', 'write'], help='Task to do.')
    parser.add_argument('--opt', type=str, default='Adam', choices=['SGD', 'Adam'],
                        help='optimizer type.')
    parser.add_argument('--logdir', default='log', type=str,
                        help='Directory for logging.', required=False)
    parser.add_argument('--loss', default='mrstft', type=str, choices=['sisdr', 'pmsqe', 'stoi', 'estoi', 'mse', 'l1', 'mrstft'],
                        help='The objective of denoising tasks.')
    parser.add_argument('--ckptdir', default='ckpt', type=str,
                        help='Path to store checkpoint result, if empty then default is used.')
    parser.add_argument(
        '--ckpt', type=str, help="Path to load target model")
    parser.add_argument('--ae_ckpt', default='PTN/DEMUCS.pth', type=str,
                        help="Path to load source pretrain model")
    parser.add_argument('--target_noise', type=str,
                        help="The path of the target noise signal to resample as running noise adaptation.")
    parser.add_argument('--cohort_list', type=str,
                        help="The path of the cohort set list to resample as running noise adaptation.")
    parser.add_argument('--eval_noise', type=str,
                        help="Changing noise path as testing")
    parser.add_argument('--out', type=str,
                        help="Path to output testing results")
    parser.add_argument('--target_type', type=str,
                        help="Noise type to adapt")

    # Options
    parser.add_argument('--seed', default=1337, type=int,
                        help='Random seed for reproducable results.', required=False)
    parser.add_argument('--alpha', default=0.5, type=float,
                        help='Choosing the alpha of resampling for noise adaptation.')
    parser.add_argument('--valid_num', default=500, type=int,
                        help='Choosing the amount of validation for noise adaptation.')
    parser.add_argument('--topk', default=250, type=int,
                        help='Choosing toppest K similar noise for noise adaptation.')
    parser.add_argument('--n_jobs', default=8, type=int,
                        help='The number of process for loading data.')
    parser.add_argument('--sample_num', default=16, type=int,
                        help='The number of demo samples to be output.')
    parser.add_argument('--device', default=2, type=int,
                        help='Assigning GPU id.')
    parser.add_argument('--metric', action='store_true',
                        help='Calculating metric scores as training.')
    parser.add_argument('--eval_init', action='store_true',
                        help='Computing initial scores before noise adaptaion.')
    parser.add_argument('--use_source_noise', action='store_true',
                        help='Choosing source noise data for noise adaptaion.')
    parser.add_argument('--empty_cache', action='store_true',
                        help='Cleaning up the memory of GPU cache in each step.')
    args = parser.parse_args()

    if args.task == 'write':
        assert not args.out is None

    torch.cuda.set_device(args.device)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args.config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

    if args.task == 'train':
        if args.method == 'PTN':
            args.ae_ckpt = None

        assert not(args.model in ['LSTM', 'GRU'] and args.loss == 'mrstft')
        if not('DAT' in args.method or 'PTN' in args.method):            
            if not args.cohort_list is None:
                assert args.use_source_noise
            if not args.use_source_noise:
                assert args.cohort_list is None
            elif not 'ALL' in args.method:
                assert not args.cohort_list is None
                with open(args.cohort_list) as f:
                    args.cohort_list = [line.strip() for line in f.readlines()][:args.topk]
            
            args.target_type = args.eval_noise.split('/')[-2]
            if not args.target_noise is None:
                args.target_noise = readfile(args.target_noise)
                
            assert not args.eval_noise is None
            args.eval_noise = readfile(args.eval_noise)
        else:
            args.target_type = args.method
    return args


def main():
    # parsing arguments
    args = argument_parsing()

    # set random seed
    set_random_seed(args.seed)
    
    if args.task == 'train':
        # set dataloader
        print(f"[DataLoder] - Applying {args.target_type} Dataset")
        train_loader = get_dataloader(args, 'train')
        dev_loader = get_dataloader(args, 'dev')

        # set model and optimizer
        init_step = 0
        if args.ckpt is None:
            print('[Model] - Building model')
            if not 'DAT' in args.method:
                if args.model in ['LSTM', 'GRU']:
                    from model import SpectrumDenoiseModel
                    model = SpectrumDenoiseModel(args)
                else:
                    from model import DenoiseModel
                    model = DenoiseModel(args)
            elif 'DAT' in args.method:
                from model import DATModel
                model = DATModel(args)

            params = model.parameters()
            optimizer = eval(f'torch.optim.{args.opt}')(params,
                                                        **args.config['optimizer'][args.opt])
        else:
            from model import load_model
            print('[Model] - Loading model parameters')
            _, model, optimizer, init_step = load_model(args)

        model.to(args.device)
        from train import train
        
        from loss import get_loss_func
        loss_func = get_loss_func(args).to(args.device)

        if args.eval_init:
            from evaluation import evaluate
            metrics = evaluate(args, dev_loader, model, loss_func, True)
            scores = ''.join([' | dev_{:} {:.5f}'.format(m, s)
                                for (m, s) in metrics[1:]])
            print('[Initial] dev_loss {:.5f}{:}'.format(metrics[0][1], scores))

        train(args, model, optimizer, train_loader,
              dev_loader, loss_func, init_step)

    elif args.task == 'test':
        from model import load_model
        from evaluation import evaluate
        import os
        os.makedirs('vcb_table/')

        loss_func = None
        for noise_type in ['ACVacuum', 'Babble', 'CafeRestaurant', 'Car', 'MetroSubway']:
            if os.path.exists(f'vcb_table/results_{noise_type}.pth'):
                results = torch.load(f'vcb_table/results_{noise_type}.pth')
            else:
                results = {}

            args.target_type = noise_type
            if not noise_type in results:
                results[noise_type] = {}
            for method in ['PTN', 'ALL', 'EXTR', 'RETV', 'GT', 'DAT_full', 'DAT_one', 'NASTAR', 'OPT']:
                if not method in results[noise_type]:           
                    results[noise_type][method] = {}
                    if 'DAT' in method:
                        args.ckpt = f'ckpt/{method}/SE_DEMUCS_20000.pth'
                    else:
                        args.ckpt = f'ckpt/{noise_type}/{method}/SE_DEMUCS_20000.pth'
                    print(f'[Model] - Loading {method} model parameters')
                    if method == 'PTN':
                        from model import DenoiseModel
                        model = DenoiseModel(args)
                    else:
                        args_ckpt, model, optimizer, init_step = load_model(args)
                    
                    model = model.to(args.device)
                    test_loader = get_dataloader(args, 'test')

                    print(
                        '[Testing] - Start testing {:} model on {:}'.format(method, args.target_type))

                    metrics = evaluate(args, test_loader, model, loss_func, True)
                    results[noise_type][method] = {m: s for (m, s) in metrics}
                    torch.save(results, f'vcb_table/results_{noise_type}.pth')

    elif args.task == 'write':
        from model import load_model
        from loss import get_loss_func
        from evaluation import write

        assert not args.out is None
        root_dir = args.out

        loss_func = None
        for noise_type in ['ACVacuum', 'Babble', 'CafeRestaurant', 'Car', 'MetroSubway']:
            args.target_type = noise_type
            for method in ['PTN', 'ALL', 'EXTR', 'RETV', 'GT', 'DAT_full', 'DAT_one', 'NASTAR', 'OPT']:
                args.ckpt = f'ckpt/{noise_type}/{method}/SE_DEMUCS_20000.pth'
                print(f'[Model] - Loading {method} model parameters')
                if method == 'PTN':
                    from model import DenoiseModel
                    model = DenoiseModel(args)
                else:
                    args_ckpt, model, optimizer, init_step = load_model(args)
                
                model = model.to(args.device)
                test_loader = get_dataloader(args, 'test')

                print(
                    '[Testing] - Start predicting {:} model on {:}'.format(method, args.target_type))
                args.out = f'{root_dir}/{noise_type}/{method}'
                write(args, test_loader, model)


if __name__ == '__main__':
    main()
