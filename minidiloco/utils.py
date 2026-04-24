import torch
import torch.distributed as dist


def setup(backend, rank, world_size):
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()
