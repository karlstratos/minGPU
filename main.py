import argparse
import os
import sys


def main(args):
    import numpy as np
    import torch
    import torch.nn as nn

    from gpu_monitor import GPUMonitor
    from logger import Logger, add_colors
    from torch.distributed import all_reduce, ReduceOp, barrier
    from torch.nn import DataParallel as DP
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from toy import check_distributed, LinearNet, MyDataset

    np.set_printoptions(formatter={'float': lambda x: '{0:0.3f}'.format(x)})
    torch.manual_seed(args.seed)

    rank, local_rank, world_size = check_distributed()
    is_main_process = local_rank in [-1, 0]
    is_distributed = world_size != -1
    logger = Logger(on=(is_main_process and not args.quiet), stamp=False)
    if is_main_process:
        gpu_monitor = GPUMonitor(logger, ['purple'])

    if is_distributed:
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
        torch.distributed.init_process_group('nccl')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.cuda.empty_cache()

    logger(f'Using device: {str(device)}', force=True)

    model = LinearNet(args.dim, args.num_labels).to(device)

    num_devices_per_process = 1
    if is_distributed:
        logger('DDP model wrapping (multi-process single-device)')

        # DDP broadcasts model weights to all processes in the world:
        # https://discuss.pytorch.org/t/when-will-dist-all-reduce-will-be-called/129918/2
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

        # lr scaling: DDP all-reduces grad info g divided by num processes (K)
        # at backward, then lets each process make its own update.
        #
        # If the update method is not gradient scale invariant, like SGD, we
        # need to use lr multiplied by K to have the original update:
        #           w' = w - lr * g = w - (K * lr) * (g/K)
        #
        # If the update method is gradient scale invariant, like any
        # AdaGrad-styleSGD method such as Adam (using epsilon=0), we should
        # NOT do that since
        #           w' = w - lr * Update(g)
        #              = w - lr * Update(g/K)
        # So multiplying lr by K will make the update K times bigger.
        # If epsilon!=0, there is no way to achieve the same update. But since
        # scale invariance is approximately preserved, we should not scale lr
        # for DDP in general if we're doing Adam and want consistency..
        #<COMMENTED>
        #scale_invariant_opts = ['adam', 'adagrad', 'rmsprop']
        #if args.opt in scale_invariant_opts: # and args.eps == 0:
        #    pass
        #else:
        #    args.lr = world_size * args.lr
        #</COMMENTED>

    elif torch.cuda.device_count() > 1:
        num_devices_per_process = len(args.gpus.split(','))
        logger(f'DP model wrapping (single-process multi-device): '
               f'using {num_devices_per_process} devices {args.gpus}')
        model = DP(model)
    else:
        logger('No model wrapping (single-process single-device)')

    dataset = MyDataset(args.num_examples, args.dim, args.num_labels)

    print_data = args.num_examples <= 10 and args.dim == 1
    if print_data:
        X = dataset.examples.numpy().transpose()
        string = f'Data ({args.num_examples}, {args.dim}): ' + \
            f'{add_colors(str(X), ["purple"])}'
        logger(string)

    if is_distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size,
                                     rank=rank, shuffle=not args.no_shuffle,
                                     seed=args.seed, drop_last=args.drop_last)
        shuffle = False  # Sampler does the shuffling
    else:
        sampler = None
        shuffle = not args.no_shuffle

    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                        shuffle=shuffle)

    # Training
    sumCE = nn.CrossEntropyLoss(reduction='sum')
    if args.opt == 'adam':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      eps=args.eps)
    elif args.opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    elif args.opt == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr,
                                        eps=args.eps)
    elif args.opt == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr,
                                        eps=args.eps)
    else:
        raise

    model.train()
    num_steps = 0
    for epoch in range(1, args.epochs + 1):
        num_steps_per_epoch = 0
        loss_sum_per_epoch = 0.
        num_correct_sum_per_epoch = 0

        if is_distributed:
            sampler.set_epoch(epoch)  # Without this, same shuffling every epoch

        # DDP w/ drop_last=False fills trailing batch with early examples.
        for batch_num, (examples, labels, indices) in enumerate(loader):
            if print_data:
                X = examples.numpy().transpose()
                string = f'Batch {batch_num} {list(examples.shape)} ' + \
                    f'({str(device)}): {add_colors(str(X), ["purple"])}'
                logger(string, force=True)

            examples = examples.to(device)
            labels = labels.to(device)
            scores, predictions = model(examples)
            loss = sumCE(scores, labels)
            num_correct = (predictions == labels).sum()
            if num_steps == 0:
                logger(f'First batch examples {list(examples.shape)} '
                       f'({str(device)})', ['yellow'])
                logger(f'First batch loss {loss:13.10g} ({str(device)})',
                       ['yellow'])

            if is_distributed:
                # Nice pictures of collective communcation:
                # https://pytorch.org/tutorials/intermediate/dist_tuto.html#collective-communication
                all_reduce(loss, op=ReduceOp.SUM)
                all_reduce(num_correct, op=ReduceOp.SUM)

                if num_steps == 0:
                    logger(f'First batch loss after all_reduce {loss:13.10g} '
                           f'({str(device)})', ['yellow'])

            loss_sum_per_epoch += loss.item()
            num_correct_sum_per_epoch += num_correct.item()

            # DDP all-reduces gradients and AVERAGES them. Each process will
            # holds G/(# processes) where G is the grad/grad^2/etc. of loss..
            loss.backward()

            if num_steps == 0:
                try:
                    g = model.grad()
                    w = model.weight()
                except AttributeError:  # DP or DDP
                    g = model.module.grad()
                    w = model.module.weight()

                logger(f'grad[0,0] after loss.backward(): {g[0, 0]:13.10g} '
                       f'({str(device)})', ['yellow'])
                logger(f'weight[0,0] before optimizer.step(): {w[0, 0]:13.10g} '
                       f'({str(device)})', ['red'])

                #<COMMENTED>
                #w2 = w[0, 0] - args.lr * g[0, 0]
                #logger(f'My own SGD-updated weight[0,0]: {w2:13.10g}'
                #       f'({str(device)})', ['darkcyan', 'bold'])
                #</COMMENTED>

            # In DDP, each process independently makes (the same) parameter
            # update using g/K. Since they all have the same init (broadcasted
            # upon DDP wrapping), the updated parameters will be the same if the
            # update is the same (e.g., SGD with scaled lr, or Adagrad with
            # eps=0 and nonscaled lr).
            optimizer.step()

            if num_steps == 0:
                try:
                    w = model.weight()
                except AttributeError:  # DP or DDP
                    w = model.module.weight()

                logger(f'weight[0,0] after optimizer.step(): {w[0, 0]:13.10g} '
                       f'({str(device)})', ['red'])

            num_steps_per_epoch += 1
            num_steps += 1
            optimizer.zero_grad()

            if is_distributed:
                barrier()

        if is_main_process:
            gpu_monitor.update()  # Print out GPU usage at each epoch

        loss_per_step = loss_sum_per_epoch / num_steps_per_epoch
        acc = num_correct_sum_per_epoch / args.num_examples * 100
        logger(f'End of epoch {epoch}: {num_steps_per_epoch} steps, '
               f'per-step loss {loss_per_step:12.8f}, acc {acc:4.2f}')

    if is_main_process:
        gpu_monitor.summarize()  # Training GPU usage summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_examples', type=int, default=100)
    parser.add_argument('--dim', type=int, default=42)
    parser.add_argument('--num_labels', type=int, default=13)
    parser.add_argument('--batch_size', type=int, default=5)  # Per process
    parser.add_argument('--opt', default='adam', type=str)
    parser.add_argument('--lr', type=float, default=1.)
    parser.add_argument('--eps', type=float, default=0)  # For scale invariance
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--drop_last', action='store_true')
    parser.add_argument('--no_shuffle', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpus', default='', type=str)
    args = parser.parse_args()

    # Set environment variables before importing libraries that use them!
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    main(args)
