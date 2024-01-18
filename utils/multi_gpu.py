import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


# [*] Packages required to import distributed data parallelism
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup_DDP(backend="nccl", verbose=False):
    """
    We don't set ADDR and PORT in here, like:
        # os.environ['MASTER_ADDR'] = 'localhost'
        # os.environ['MASTER_PORT'] = '12355'
    Because program's ADDR and PORT can be given automatically at startup.
    E.g. You can set ADDR and PORT by using:
        python -m torch.distributed.launch --master_addr="192.168.1.201" --master_port=23456 ...
    You don't set rank and world_size in dist.init_process_group() explicitly.
    :param backend:
    :param verbose:
    :return:
    """
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    # If the OS is Windows or macOS, use gloo instead of nccl
    dist.init_process_group(backend=backend)
    # set distributed device
    device = torch.device("cuda:{}".format(local_rank))
    if verbose:
        print("Using device: {}".format(device))
        print(f"local rank: {local_rank}, global rank: {rank}, world size: {world_size}")
    return rank, local_rank, world_size, device