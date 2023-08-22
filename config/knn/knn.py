import argparse
import os


def knn():
    args = argparse.Namespace()

    args.dataset = 'CUB'
    args.arch = 'vit-small'
    args.pretrained_weights = ''
    args.output_dir = './out'
    args.seed = 7

    if args.dataset == 'imagenet1k':
        args.num_workers = 12
        args.prefetch_factor = 3
        args.pin_memory = True
        args.patch_size = 16
        args.input_size = 224
        args.data_root = '/path/to/ILSVRC2012'
        args.batch_size_per_gpu = 1024
    elif args.dataset == 'cifar100':
        args.num_workers = 4
        args.prefetch_factor = 2
        args.pin_memory = True
        args.patch_size = 2
        args.input_size = 32
        args.data_root = '/path/to/dataset/cifar100'
        args.batch_size_per_gpu = 256
    elif args.dataset == 'CUB':
        args.num_workers = 12
        args.prefetch_factor = 3
        args.pin_memory = True
        args.patch_size = 16
        args.input_size = 224
        args.data_root = '/public/datasets/CUB-200-2011/CUB_200_2011/splited_images'
        args.batch_size_per_gpu = 512
    elif args.dataset == 'Stanford_Dogs':
        args.num_workers = 16
        args.prefetch_factor = 3
        args.pin_memory = True
        args.patch_size = 16
        args.input_size = 224
        args.data_root = '/public/datasets/Stanford_Dogs/splited_images/'
        args.batch_size_per_gpu = 256
    else:
        args.num_workers = 4
        args.prefetch_factor = 2
        args.pin_memory = True
        args.patch_size = 2
        args.input_size = 32
        args.data_root = '/path/to/dataset/cifar10'
        args.batch_size_per_gpu = 256

    args.encoder = 'momentum_encoder'
    args.nb_knn = [10, 20, 100, 200]
    args.temperature = 0.07
    args.use_cuda = True
    args.dump_features = None
    args.load_features = None

    # ----------------#
    args.dist_url = 'tcp://localhost:12621'
    args.dist_backend = 'nccl'
    args.rank = 0
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    args.world_size = 1

    args.exclude_file_list = ['__pycache__', '.vscode', 'log', 'ckpt', '.git', 'out', 'dataset', 'weight']

    return args
