# Inspired by/adapted from:
# https://github.com/pytorch/workshops/blob/master/FSDP_Workshop/performance/gpu_memory.py
#
# As recommended, max out batch size, but make sure there's no cudaMalloc retry.

import torch


GB_SIZE = 1073741824  # 1024^3
MB_SIZE = 1048576     # 1024^2


def B2G(num_bytes, precision=2):
    num_gigabytes = round(num_bytes / GB_SIZE, ndigits=precision)
    return num_gigabytes


class GPUMonitor:
    """Typically used on rank 0 only as a representative GPU"""

    def __init__(self, logger, colors=[]):
        self.log = lambda string: logger(string, colors)

        current_free, full_gpu_mem = torch.cuda.mem_get_info()
        self.total_mem = B2G(full_gpu_mem)

        self.log(f'Total memory/GPU: {self.total_mem}G '
                 f'({B2G(current_free)}G currently free)')
        self.start_tracking()

    def start_tracking(self):
        torch.cuda.reset_peak_memory_stats()

        self.used_mems = []
        self.used_percs = []
        self.num_retries = 0
        self.total_ooms = 0
        self.max_reserved = 0

        self.log('GPU memory tracking stats reset')

    def update(self, log=False):
        """Take a snapshot of the current GPU memory usage (e.g., at the end of
        each epoch)."""

        # memory_reserved() gives the total memory footprint, including cache
        # allocation. If you want the actual memory usage by tensors, use
        # memory_allocated().
        used_mem = B2G(torch.cuda.memory_reserved())
        used_perc = round(100 * used_mem / self.total_mem, 2)

        self.used_mems.append(used_mem)
        self.used_percs.append(used_perc)

        self.log(f'Current GPU memory usage: {used_mem}G ({used_perc}%)')

    def summarize(self):
        self.log(f'used_mems: {self.used_mems}')
        self.log(f'used_percs: {self.used_percs}')

        # Get the peak cached memory since the beginning:
        # https://pytorch.org/docs/stable/generated/torch.cuda.max_memory_reserved.html#torch.cuda.max_memory_reserved
        max_used_mem = B2G(torch.cuda.max_memory_reserved())
        max_used_perc = round(100 * max_used_mem / self.total_mem, 2)

        self.log(f'max_used_mem: {max_used_mem}G')
        self.log(f'max_used_perc: {max_used_perc}%')

        mem_stats = torch.cuda.memory_stats()
        self.num_malloc_retries = mem_stats.get('num_alloc_retries', 0)
        self.num_ooms = mem_stats.get('num_ooms', 0)

        self.log(f'# cudaMalloc retries: {self.num_malloc_retries}')
        self.log(f'# out-of-memory errors thrown: {self.num_ooms}')
        if self.num_malloc_retries > 0:
            self.log('You should decrease batch size until there is no '
                     'cudaMaloc retry!')
