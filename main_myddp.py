# python main_myddp.py  --dist_method none --batch_size 12
# python main_myddp.py  --dist_method manual --batch_size 12
# torchrun --standalone --nnodes=1 --nproc_per_node=2 main_myddp.py --dist_method torch --batch_size 12
#
# python main_myddp.py  --dist_method manual --batch_size 4
# torchrun --standalone --nnodes=1 --nproc_per_node=2 main_myddp.py --dist_method torch --batch_size 4

import argparse
import os
import torch
import torch.multiprocessing as mp
import torch.nn as nn

from logger import Logger
from torch.utils.data import DataLoader
from toy import check_distributed, LinearNet, MyDataset


os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
EXAMPLES = [[-1.2957, -0.2762],
            [-0.9113, -0.4733],
            [ 1.8040,  0.7224],
            [-0.1713,  0.8524],
            [-1.9583, -0.4739],
            [ 0.6777,  1.2174],
            [ 0.0669, -0.7058],
            [ 0.1843, -0.9769],
            [-0.6657, -0.5356],
            [-0.6752, -1.1082],
            [-0.3811, -0.9390],
            [ 2.5346, -0.0949]]
LABELS = [0, 2, 0, 0, 1, 2, 2, 2, 2, 1, 2, 2]
WORLD_SIZE = 2
LR=0.1
sumCE = nn.CrossEntropyLoss(reduction='sum')

def str_wij(model, device, i=1, j=1):
    wij = model.weight()[i, j].item()
    return f'w({i},{j}) = {wij:12.10f} ({str(device)})'

def str_gij(model, device, i=1, j=1):
    gij = model.grad()[i, j].item()
    return f'g({i},{j}) = {gij:12.10f} ({str(device)})'


def main(batch_size, epochs):
    logger = Logger(stamp=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger(f'Using device: {str(device)}', force=True)

    torch.manual_seed(42)
    model = LinearNet(2, 3).to(device)
    logger(f'Initial {str_wij(model, device)}', ['green'], force=True)

    dataset = MyDataset(to_load=(EXAMPLES, LABELS))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    model.train()
    num_steps = 0
    for epoch in range(1, epochs + 1):
        num_steps_per_epoch = 0
        loss_sum = 0.

        for batch_num, (examples, labels, indices) in enumerate(loader):
            if num_steps == 0:
                logger(f'Batch {batch_num}: {examples.tolist()}', ['blue'],
                       force=True)
            examples = examples.to(device)
            labels = labels.to(device)
            scores, predictions = model(examples)
            loss = sumCE(scores, labels)
            loss_sum += loss.item()

            loss.backward()
            optimizer.step()
            if num_steps == 0:
                logger(f'{str_gij(model, device)}', ['yellow'], force=True)
                logger(f'{str_wij(model, device)}', ['green'], force=True)
            num_steps_per_epoch += 1
            num_steps += 1
            optimizer.zero_grad()

        lossps = loss_sum / num_steps_per_epoch
        logger(f'End of epoch {epoch}: {num_steps_per_epoch} steps, '
               f'per-step loss {lossps:12.8f}')


def main_torch(batch_size, epochs):
    from torch.distributed import (all_reduce, ReduceOp, barrier,
                                   init_process_group, destroy_process_group)
    from torch.utils.data.distributed import DistributedSampler
    from torch.nn.parallel import DistributedDataParallel as DDP

    rank, local_rank, world_size = check_distributed()
    is_main_process = local_rank in [-1, 0]
    is_distributed = world_size != -1
    assert is_distributed
    assert rank == local_rank
    logger = Logger(on=is_main_process, stamp=False)
    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)
    init_process_group('nccl', rank=rank, world_size=WORLD_SIZE)
    logger(f'Using device: {str(device)}', force=True)

    torch.manual_seed(42)
    model = LinearNet(2, 3).to(device)
    model = DDP(model, device_ids=[rank], output_device=rank)
    logger(f'Initial {str_wij(model.module, device)}', ['green'], force=True)

    dataset = MyDataset(to_load=(EXAMPLES, LABELS))
    sampler = DistributedSampler(dataset, num_replicas=WORLD_SIZE,
                                 rank=rank, shuffle=False, seed=42)
    loader = DataLoader(dataset, batch_size=batch_size // WORLD_SIZE,
                        sampler=sampler, shuffle=False)

    optimizer = torch.optim.SGD(model.parameters(), lr=WORLD_SIZE * LR)
    model.train()
    num_steps = 0
    for epoch in range(1, epochs + 1):
        num_steps_per_epoch = 0
        loss_sum = 0.

        if is_distributed:
            sampler.set_epoch(epoch)  # No shuffling, but formality

        for batch_num, (examples, labels, indices) in enumerate(loader):
            if num_steps == 0:
                logger(f'Batch {batch_num} ({str(device)}): '
                       f'{examples.tolist()}', ['blue'], force=True)
            examples = examples.to(device)
            labels = labels.to(device)
            scores, predictions = model(examples)
            loss = sumCE(scores, labels)
            all_reduce(loss, op=ReduceOp.SUM)
            loss_sum += loss.item()

            loss.backward()
            optimizer.step()
            if num_steps == 0:
                logger(f'{str_gij(model.module, device)}', ['yellow'], force=True)
                logger(f'{str_wij(model.module, device)}', ['green'], force=True)
            num_steps_per_epoch += 1
            num_steps += 1
            optimizer.zero_grad()

        lossps = loss_sum / num_steps_per_epoch
        logger(f'End of epoch {epoch}: {num_steps_per_epoch} steps, '
               f'per-step loss {lossps:12.8f}')


def run_worker(rank, batch_size, epochs):
    from torch.distributed import (all_reduce, ReduceOp, barrier,
                                   init_process_group, destroy_process_group)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Rank and world_size can be set as env variables too.
    init_process_group('nccl', rank=rank, world_size=WORLD_SIZE)

    logger = Logger(on=(rank == 0), stamp=False)
    device = torch.device('cuda', rank)
    torch.cuda.set_device(rank)
    logger(f'Using device: {str(device)}', force=True)

    torch.manual_seed(42)
    model = LinearNet(2, 3).to(device)
    logger(f'Initial {str_wij(model, device)}', ['green'], force=True)

    num_examples_per_process = len(EXAMPLES) // WORLD_SIZE
    indices = list(range(rank, len(EXAMPLES), WORLD_SIZE))
    print(indices)
    dataset = MyDataset(to_load=([EXAMPLES[i] for i in indices],
                                 [LABELS[i] for i in indices]))
    loader = DataLoader(dataset, batch_size=batch_size // WORLD_SIZE,
                        shuffle=False)

    optimizer = torch.optim.SGD(model.parameters(), lr=WORLD_SIZE * LR)
    model.train()
    num_steps = 0
    for epoch in range(1, epochs + 1):
        num_steps_per_epoch = 0
        loss_sum = 0.

        for batch_num, (examples, labels, indices) in enumerate(loader):
            if num_steps == 0:
                logger(f'Batch {batch_num} ({str(device)}): '
                       f'{examples.tolist()}', ['blue'], force=True)
            examples = examples.to(device)
            labels = labels.to(device)
            scores, predictions = model(examples)
            loss = sumCE(scores, labels)

            all_reduce(loss, op=ReduceOp.SUM)

            loss_sum += loss.item()

            loss.backward()

            # Manually average gradients across processes
            for param in model.parameters():
                all_reduce(param.grad.data, op=ReduceOp.SUM)
                param.grad.data /= WORLD_SIZE

            barrier()
            optimizer.step()
            if num_steps == 0:
                logger(f'{str_gij(model, device)}', ['yellow'], force=True)
                logger(f'{str_wij(model, device)}', ['green'], force=True)
            num_steps_per_epoch += 1
            num_steps += 1
            optimizer.zero_grad()

        lossps = loss_sum / num_steps_per_epoch
        logger(f'End of epoch {epoch}: {num_steps_per_epoch} steps, '
               f'per-step loss {lossps:12.8f}')

    # Clean up
    destroy_process_group()


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist_method', type=str, default='none',
                        choices=['none', 'torch', 'manual'])
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=3)
    args = parser.parse_args()
    if args.dist_method == 'none':
        main(args.batch_size, args.epochs)
    elif args.dist_method == 'torch':
        main_torch(args.batch_size, args.epochs)
    else:
        assert args.dist_method == 'manual'
        mp.spawn(run_worker, args=(args.batch_size, args.epochs),
                 nprocs=WORLD_SIZE, join=True)
