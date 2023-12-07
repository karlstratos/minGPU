import argparse
import os
import sys


def main(args):
    import torch
    import torch.nn as nn

    from logger import Logger
    from torch.distributed import all_reduce, ReduceOp
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.nn import DataParallel as DP
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from toy import check_distributed, MyModel, MyDataset


    torch.manual_seed(args.seed)

    rank, local_rank, world_size = check_distributed()
    is_main_process = local_rank in [-1, 0]
    is_distributed = world_size != -1
    logger = Logger(on=is_main_process)

    if is_distributed:
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
        torch.distributed.init_process_group('nccl')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger(f'Using device: {str(device)}', force=True)  # Print each process

    model = MyModel(args.dim, args.num_labels).to(device)

    num_devices_per_process = 1
    if is_distributed:
        logger('DDP model wrapping (multi-process single-device)')
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    elif torch.cuda.device_count() > 1:
        num_devices_per_process = len(args.gpus.split(','))
        logger(f'DP model wrapping (single-process multi-device): '
               f'using {num_devices_per_process} devices {args.gpus}')
        model = DP(model)
    else:
        logger('No model wrapping (single-process single-device)')

    dataset = MyDataset(args.num_examples, args.dim, args.num_labels)
    if is_distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size,
                                     rank=rank, shuffle=not args.no_shuffle,
                                     seed=args.seed, drop_last=False)
        shuffle = False  # Sampler does the shuffling
    else:
        sampler = None
        shuffle = not args.no_shuffle

    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                        shuffle=shuffle)

    # Training
    sumCE = nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    for epoch in range(1, args.epochs + 1):
        num_steps = 0
        loss_sum = 0.
        num_correct_sum = 0

        # DDP w/ drop_last=False: Fill trailing batch with early examples.
        for batch_num, (examples, labels) in enumerate(loader):
            examples = examples.to(device)
            labels = labels.to(device)
            scores, predictions = model(examples)
            loss = sumCE(scores, labels)
            num_correct = (predictions == labels).sum()

            if num_devices_per_process > 1:  # DP returns a vector of losses
                loss = loss.sum()

            if is_distributed:
                all_reduce(loss, op=ReduceOp.SUM)
                all_reduce(num_correct, op=ReduceOp.SUM)

            loss_sum += loss.item()
            num_correct_sum += num_correct.item()

            # DDP: all_reduce(gradients, MEAN)
            loss.backward()

            # DDP: all_reduce(gradients, SUM) before updating
            optimizer.step()

            num_steps += 1
            optimizer.zero_grad()

        loss_per_step = loss_sum / num_steps
        acc = num_correct_sum / args.num_examples * 100
        logger(f'End of epoch {epoch}:  per-step loss {loss_per_step:8.4f},'
               f'  acc {acc:4.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_examples', type=int, default=100)
    parser.add_argument('--dim', type=int, default=42)
    parser.add_argument('--num_labels', type=int, default=13)
    parser.add_argument('--batch_size', type=int, default=5)  # Per process
    parser.add_argument('--lr', type=float, default=1.)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--no_shuffle', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpus', default='', type=str)
    args = parser.parse_args()

    # Set environment variables before importing libraries that use them!
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    main(args)
