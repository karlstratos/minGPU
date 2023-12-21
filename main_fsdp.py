import argparse
import functools
import os
import sys


def main(args):
    import numpy as np
    import torch
    import torch.nn as nn

    from gpu_monitor import GPUMonitor
    from logger import Logger, add_colors
    from torch.distributed import all_reduce, ReduceOp, barrier
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        StateDictType,
        FullStateDictConfig,
        FullOptimStateDictConfig,
        BackwardPrefetch,
        ShardingStrategy,
        CPUOffload,
    )
    from torch.distributed.fsdp.wrap import (
        size_based_auto_wrap_policy, enable_wrap, wrap
    )
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from toy import check_distributed, ThreeLayerNet, MyDataset, param_str


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

    model = ThreeLayerNet().to(device)
    model.init()
    logger(model.weight_str(), ['yellow'])

    is_ddp = (is_distributed and args.dist == 'ddp')
    is_fsdp = (is_distributed and args.dist == 'fsdp')

    if is_ddp:
        logger('DDP model wrapping (multi-process single-device)')
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)


    def wrap_fsdp(model, sync_module_states=False):
        """
        sync_module_states=True syncs loaded checkpoint states from rank 0 to
        rest of the world. Not needed for training when all the world is
        initialized the same anyway. Needed for loading.
        """

        # https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.ShardingStrategy
        strategy = {
            'full': ShardingStrategy.FULL_SHARD,  # ~ ZeRO Level 3
            'grad_op': ShardingStrategy.SHARD_GRAD_OP,  # ~ ZeRO Level 2
            'no_shard': ShardingStrategy.NO_SHARD,  # = DDP
            'hybrid': ShardingStrategy.HYBRID_SHARD  # full shard within a node
        }[args.strategy]

        shared_fsdp_kwargs = dict(
            sharding_strategy=strategy,
            cpu_offload=CPUOffload(offload_params=False),
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=rank,
            ignored_modules=None,
            limit_all_gathers=False,
            use_orig_params=args.use_orig_params,
            sync_module_states=sync_module_states
        )

        if args.wrap == 'naive':
            model = FSDP(model, **shared_fsdp_kwargs)
        elif args.wrap == 'size':
            # Wrap a submodule iff # params > 7 => layer1/layer2 not wrapped
            shared_fsdp_kwargs['auto_wrap_policy'] = functools.partial(
                size_based_auto_wrap_policy, min_num_params=7)
            model = FSDP(model, **shared_fsdp_kwargs)
        elif args.wrap == 'layer':  # Manual layer-wise sharding
            # enable_wrap/wrap is useful for applying the same config to all
            # child modules you wrap. Alternatively, you could wrap explicitly
            # like this:
            #model.layer1 = FSDP(model.layer1, **shared_fsdp_kwargs)
            #model.layer2 = FSDP(model.layer2, **shared_fsdp_kwargs)
            #model.layer3 = FSDP(model.layer3, **shared_fsdp_kwargs)
            with enable_wrap(wrapper_cls=FSDP, **shared_fsdp_kwargs):
                model.layer1 = wrap(model.layer1)
                model.layer2 = wrap(model.layer2)
                model.layer3 = wrap(model.layer3)

            # Always wrap the top module.
            # https://github.com/pytorch/xla/blob/bf5edb3938245b0d125f222b2a07ac0835a90081/test/test_train_mp_mnist_fsdp_with_ckpt.py#L216C13-L216C13
            model = FSDP(model, **shared_fsdp_kwargs)
        else:
            raise

        return model


    if is_fsdp:
        logger('FSDP model wrapping (multi-process single-device)')
        model = wrap_fsdp(model)

    logger(f'Model type: {str(type(model))}', ['green'])

    if is_fsdp:
        # How are the three child submodules wrapped inside FSDP model?
        logger(f'layer1 type: {str(type(model.layer1))}', ['green'])
        logger(f'layer2 type: {str(type(model.layer2))}', ['green'])
        logger(f'layer3 type: {str(type(model.layer3))}', ['green'])

        logger('Displaying sharded params', ['underline'])
        barrier()
        if args.wrap == 'naive':
            # The whole model is sharded.
            logger(f'model._flat_param ({str(device)}): '
                   f'{param_str(model._flat_param)}\n', ['yellow'], force=True)

        elif args.wrap == 'size':
            # Layer 3 is sharded.. The rest (layer 1,2) is sharded in model.
            logger(f'model._flat_param ({str(device)}): '
                   f'{param_str(model._flat_param)}\n'
                   f'model.layer3._flat_param ({str(device)}): '
                   f'{param_str(model.layer3._flat_param)}\n', ['yellow'],
                   force=True)

        elif args.wrap == 'layer':
            # Layer 1,2,3 are sharded separately.
            logger(f'model.layer1._flat_param ({str(device)}): '
                   f'{param_str(model.layer1._flat_param)}\n'
                   f'model.layer2._flat_param ({str(device)}): '
                   f'{param_str(model.layer2._flat_param)}\n'
                   f'model.layer3._flat_param ({str(device)}): '
                   f'{param_str(model.layer3._flat_param)}\n', ['yellow'],
                   force=True)

    dataset = MyDataset(args.num_examples, 2, 3)
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=0)

    def do_epoch(model, optimizer=None):
        num_steps = 0
        loss_sum = 0.
        num_correct_sum = 0

        if is_distributed:
            sampler.set_epoch(epoch)  # Without this, same shuffling every epoch

        for batch_num, (examples, labels, indices) in enumerate(loader):

            logger(str(indices) + f' batch {batch_num} ({str(device)})', force=True)

            examples = examples.to(device)
            labels = labels.to(device)
            scores, predictions = model(examples)
            loss = sumCE(scores, labels)
            num_correct = (predictions == labels).sum()

            if is_distributed:
                all_reduce(loss, op=ReduceOp.SUM)
                all_reduce(num_correct, op=ReduceOp.SUM)

            loss_sum += loss.item()
            num_correct_sum += num_correct.item()

            if optimizer is not None:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            num_steps += 1

            if is_distributed:
                barrier()

        if is_main_process:
            gpu_monitor.update()  # Print out GPU usage at each epoch

        lossps = loss_sum / num_steps
        acc = num_correct_sum / args.num_examples * 100
        return num_steps, lossps, acc

    model.train()
    for epoch in range(1, args.epochs + 1):
        num_steps, lossps, acc  = do_epoch(model, optimizer)
        logger(f'End of epoch {epoch}: {num_steps} steps, '
               f'per-step loss {lossps:12.8f}, acc {acc:4.2f}')

    with torch.no_grad():
        _, lossps, _ = do_epoch(model)
    logger(f'Final per-step loss: {lossps:12.8f}', ['red'])

    if is_main_process:
        gpu_monitor.summarize()  # Training GPU usage summary

    logger(f'Saving model/optimizer ({str(device)})')
    if (not is_distributed) or is_ddp:
        if is_main_process:  # Just need to save on rank 0
            model_state = model.module.state_dict() if is_ddp else \
                model.state_dict()
            optim_state = optimizer.state_dict()

    if is_fsdp:
        # https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullStateDictConfig
        sd_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        osd_cfg = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, sd_cfg,
                                  osd_cfg):
            # CPU tensors on rank 0, empty otherwise
            model_state = model.state_dict()

        # Consolidates the full optimizer state on rank 0 (still needs to be
        # called on all ranks):
        # https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.full_optim_state_dict
        optim_state = FSDP.full_optim_state_dict(model, optimizer,
                                                 rank0_only=True)

    if is_main_process:
        torch.save(model_state, 'model.pt')
        torch.save(optim_state, 'optim.pt')

    logger(f'Loading saved states to model2/optimizer2 ({str(device)})')
    model2 = ThreeLayerNet().to(device)
    model2.init(delta=1)

    if is_main_process:
        model_state2 = torch.load('model.pt')
        model2.load_state_dict(model_state2)

    if is_ddp:
        model2 = DDP(model2, device_ids=[local_rank], output_device=local_rank)

    if is_fsdp:  # Must sync module states from rank 0 to the world!
        model2 = wrap_fsdp(model2, sync_module_states=True)

    model2.eval()
    with torch.no_grad():
        _, lossps, _ = do_epoch(model2)
    logger(f'Loaded per-step loss: {lossps:12.8f}', ['red'])

    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=42., eps=0)

    if (not is_distributed) or is_ddp:  # Must load optim states on all ranks
        optim_state2 = torch.load('optim.pt')
        optimizer2.load_state_dict(optim_state2)

    if is_fsdp:
        # Only load optim states on rank 0
        optim_state2 = torch.load('optim.pt') if is_main_process else None

        # But call this from all ranks, though only rank 0 has non-None osd.
        # (scatter_... uses less aggregate CPU memory than shard_... which
        # requires every rank to have full osd, see
        # https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.scatter_full_optim_state_dict)
        sharded_osd = FSDP.scatter_full_optim_state_dict(optim_state2, model2)
        optimizer2.load_state_dict(sharded_osd)

    logger(f'Continuing training', ['darkcyan'])
    model2.train()
    for epoch in range(args.epochs + 1, 2 * args.epochs + 1):
        num_steps, lossps, acc  = do_epoch(model2, optimizer2)
        logger(f'End of epoch {epoch}: {num_steps} steps, '
               f'per-step loss {lossps:12.8f}, acc {acc:4.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_examples', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=5)  # Per process
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--drop_last', action='store_true')
    parser.add_argument('--no_shuffle', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--dist', type=str, default='fsdp',
                        choices=['ddp', 'fsdp'])
    parser.add_argument('--use_orig_params', action='store_true',
                        help='if True, e.g., can use named_parameters()')
    parser.add_argument('--min_num_params', type=int, default=7)
    parser.add_argument('--wrap', type=str, default='naive',
                        choices=['naive', 'size', 'layer'])
    parser.add_argument('--strategy', type=str, default='full',
                        choices=['full', 'grad_op', 'no_shard', 'hybrid'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpus', default='', type=str)
    args = parser.parse_args()

    # Set environment variables before importing libraries that use them!
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    main(args)
