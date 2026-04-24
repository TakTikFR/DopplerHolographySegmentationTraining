import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from MiniDiLoCo.aggregator import AllReduce

class Trainer:
    """ Generic distributed PyTorch trainer """

    def __init__(self, rank, world_size, model, dataloader):
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{rank}")
        self.model = DDP(model.to(self.device), device_ids=[rank])
        self.dataloader = dataloader


    def train(self, strategy, total_steps):
        """ Distributed outer training loop.

        Args:
            strategy (Strategy): Distributed strategy chooses who performs the inner loop
            total_steps (Int): Number of outer loops
        """        

        strategy._init_node(self.model, self.rank, self.world_size, total_steps)
        all_reduce = AllReduce(strategy.outer_opt, strategy.inner_opt)
        
        self.model.train()

        for outer_step in range(total_steps):
            self.dataloader.sampler.set_epoch(outer_step)

            running_loss = 0.0
            for inner_step, batch in enumerate(self.dataloader):
                if inner_step >= strategy.H:
                    break

                if isinstance(batch, dict):
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                else:
                    imgs, masks = batch
                    batch = {"pixel_values": imgs.to(self.device), "labels": masks.to(self.device)}

                # Inner loop
                loss = strategy.step(batch)
                running_loss += loss

            # Parameters synchronisation
            all_reduce.aggregate()

            strategy.outer_opt.zero_grad()
            strategy.outer_opt.step()

            print(f"Rank: {self.rank} - Outer step: {outer_step} - Average loss: {running_loss / strategy.H}")

import numpy as np
import torch
from torchvision import transforms

# ── À définir au niveau MODULE (top-level), PAS dans diloco_worker ──────────
_to_tensor = transforms.ToTensor()

def _seg_collate_fn(batch):
    imgs, masks = zip(*batch)
    imgs  = torch.stack([_to_tensor(x) if not isinstance(x, torch.Tensor)
                         else x for x in imgs])
    masks = torch.stack([torch.from_numpy(np.array(m)).float()
                         if not isinstance(m, torch.Tensor)
                         else m for m in masks])
    return imgs, masks
# ────────────────────────────────────────────────────────────────────────────


def diloco_worker(rank, world_size, model, dataset, batch_size, loss_fn, num_epochs, lr, save_path):
    import os, sys
    from torch.utils.data import DataLoader, DistributedSampler

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['LOCAL_RANK']  = str(rank)
    os.environ['WORLD_SIZE']  = str(world_size)

    sys.path.insert(0, os.path.dirname(__file__))
    from utils import setup, cleanup
    from strategy import Diloco
    import torch.distributed as dist

    setup('nccl', rank, world_size)

    class ModelOutput:
        def __init__(self, logits):
            self.logits = logits  # ← logits, pas loss

    class SegWrapper(torch.nn.Module):
        def __init__(self, m, fn):
            super().__init__()
            self.m = m
            self.fn = fn

        def forward(self, pixel_values, labels, **kw):
            logits = self.m(pixel_values)  # sortie brute du modèle
            return ModelOutput(logits=logits)  # strategy.py calcule la loss lui-même

        def parameters(self, recurse=True):
            return self.m.parameters(recurse)

    wrapped = SegWrapper(model, loss_fn)
    # ─────────────────────────────────────────────────────────────────────────

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, sampler=sampler,
        num_workers=2, collate_fn=_seg_collate_fn,
    )

    strategy = Diloco(inner_lr=lr, loss_fn=loss_fn)
    trainer  = Trainer(rank, world_size, wrapped, dataloader)   # ← wrapped
    trainer.train(strategy, total_steps=num_epochs)
    
    # ── Sauvegarde depuis rank 0 uniquement ──────────────────────────────────
    if rank == 0:
        torch.save(wrapped.m.state_dict(), save_path)
        print(f"Model saved to {save_path}")
    # ────────────────────────────────────────────────────────────────────────

    dist.barrier()
    cleanup()