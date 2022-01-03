import argparse
import logging
import os


def main(args):
    import torch

    from util import Logger, set_seed, check_distributed

    set_seed()
    rank, local_rank, world_size = check_distributed()
    logger = Logger(local_rank in [-1, 0])
    logger.log(str(args))

    if local_rank == -1:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
        dist.init_process_group('nccl')
    logger.log(str(device))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_examples', type=int, default=5)
    parser.add_argument('--dim', type=int, default=2)
    parser.add_argument('--num_labels', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--gpus', default='', type=str)
    args = parser.parse_args()

    # Set environment variables before importing libraries that use them!
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    main(args)
