#!/usr/bin/env python 
import builtins
import logging

import torch as t
import torch.distributed as dist
from torch.nn.functional import mse_loss
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.models import resnet50
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm import tqdm


# 创建Dataset时，尽可能减小CPU负担，这样测试性能时可以保证是GPU的计算能力/通信为主要瓶颈。 
class DummyDataset(Dataset):
    def __getitem__(self, item):
        img = t.rand(3, 224, 224)
        label = int(img.mean() < 0.5)
        return img, label
    def __len__(self):
        return 1000


# 判断是否为主进程 
def is_master_proc(num_gpus=8):
    if t.distributed.is_initialized():
        return dist.get_rank() % num_gpus == 0
    else:
        return True


def setup_logging():
    if is_master_proc():
        logging.basicConfig(filename='test.log', filemode='w', level=logging.INFO)
        logging.getLogger().addHandler(logging.StreamHandler())
        logging.info('logging started.')
    else:
        def print_none(*args, **kwargs):
            pass
        # 将内置print函数变为一个空函数，从而使非主进程的进程不会输出。          
        builtins.print = print_none


def train(cfg):
    setup_logging()
    dataset = DummyDataset()
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, cfg.batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    model = resnet50(num_classes=1).cuda()
    # 因为每个进程初始化时已经指定了GPU，所以此处不需要明确指出需要将模型迁移到哪个GPU。      
    cur_device = t.cuda.current_device()
    if cfg.num_gpus > 1:
        # 使用DDP包裹模型，可以自动帮我们同步参数。          
        model = DDP(model, device_ids=[cur_device], output_device=cur_device)
    optim = SGD(model.parameters(), 0.001)
    # 如果用到tqdm，需要在非主进程的进程进行抑制。      
    for i, (data, label) in tqdm(enumerate(dataloader), disable=not is_master_proc()):
        data = data.to(cur_device)
        label = label.to(t.float32).reshape(-1, 1).to(cur_device)
        pred = model(data)
        optim.zero_grad()
        loss = mse_loss(pred, label)
        loss.backward()
        optim.step()
    logging.info('finish train for one epoch.')
    logging.info(f'last loss before reduce:{loss.item()}')
    dist.all_reduce(loss)
    # 可以看到，单卡loss和其他卡上的loss不同，如果涉及到计算准确率等，需要先同步其他卡的结果，然后进行统计。      
    logging.info(f'last loss after reduce:{loss.item()}')


def run(local_rank, func, cfg):
    t.distributed.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:9999',
        world_size=cfg.num_gpus,
        rank=local_rank,
    )

    t.cuda.set_device(local_rank)
    func(cfg)


def launch_job(cfg, func, daemon=False):
    if cfg.num_gpus > 1:
        t.multiprocessing.spawn(
            run,
            nprocs=cfg.num_gpus,
            args=(
                func,
                cfg,
            ),
            daemon=daemon,
        )
    else:
        func(cfg)


class Cfg:
    def __init__(self):
        self.num_gpus = 2
        self.batch_size = 12


if __name__ == '__main__':
    cfg = Cfg()
    launch_job(cfg, train)
