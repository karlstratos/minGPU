import argparse
import logging
import os


def main(args):
    import torch

    from data import MyDataset
    from model import MyModel
    from util import Logger, set_seed, check_distributed

    set_seed(args.seed)
    rank, local_rank, world_size = check_distributed()
    is_main_process = local_rank in [-1, 0]
    is_distributed = local_rank != -1

    logger = Logger(on=is_main_process)
    logger.log(str(args))
    logger.log(f'rank {rank} local_rank {local_rank} world_size {world_size}',
               force=True)

    if local_rank == -1:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
        torch.distributed.init_process_group('nccl')

    logger.log(f'Using device: {str(device)}', force=True)

    model = MyModel(args.dim, args.num_labels).to(device)

    num_devices_per_process = 1
    if is_distributed:
        logger.log('DDP wrapping')
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank)
    elif torch.cuda.device_count() > 1:
        num_devices_per_process = len(args.gpus.split(','))
        logger.log(f'DP model wrapping, {num_devices_per_process} devices: '
                   f'{args.gpus}')
        model = torch.nn.DataParallel(model)
    else:
        logger.log('Single-process single-device, no model wrapping')

    dataset = MyDataset(args.num_examples, args.dim, args.num_labels)
    if is_distributed:
        assert args.batch_size % world_size == 0
        batch_size_per_process = args.batch_size // world_size
        logger.log(f'Using DistributedSampler in DataLoader, batch size '
                   f'{batch_size_per_process} per process')
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank,
            shuffle=not args.no_shuffle, seed=args.seed, drop_last=False)
        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size_per_process,
                                             sampler=sampler)
    else:
        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch_size,
                                             shuffle=not args.no_shuffle)

    # Training
    sumCE = torch.nn.CrossEntropyLoss(reduction='sum')
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
                torch.distributed.all_reduce(loss,
                                             op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(num_correct,
                                             op=torch.distributed.ReduceOp.SUM)

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
        logger.log(f'End of epoch {epoch}:  per-step loss {loss_per_step:8.4f},'
                   f'  acc {acc:4.2f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_examples', type=int, default=5)
    parser.add_argument('--dim', type=int, default=2)
    parser.add_argument('--num_labels', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1.)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--no_shuffle', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpus', default='', type=str)
    args = parser.parse_args()

    # Set environment variables before importing libraries that use them!
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    main(args)
