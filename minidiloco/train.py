import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from aggregator import AllReduce


class Trainer:
    """Generic distributed PyTorch trainer."""

    def __init__(self, rank, world_size, model, dataloader):
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{rank}")
        self.model = DDP(model.to(self.device), device_ids=[rank])
        self.dataloader = dataloader

    def train(self, strategy, total_steps):
        """Distributed outer training loop.

        Args:
            strategy (Strategy): Provides the inner-loop logic.
            total_steps (int): Number of outer synchronisation rounds.

        Returns:
            list[float]: Average loss per outer step.
        """
        strategy._init_node(self.model, self.rank, self.world_size, total_steps)
        all_reduce = AllReduce(strategy.outer_opt, strategy.inner_opt)

        self.model.train()
        history = []

        for outer_step in range(total_steps):
            self.dataloader.sampler.set_epoch(outer_step)
            running_loss = 0.0

            for inner_step, batch in enumerate(self.dataloader):
                if inner_step >= strategy.H:
                    break
                batch = {k: v.to(self.device) for k, v in batch.items()}
                running_loss += strategy.step(batch)

            all_reduce.aggregate()
            strategy.outer_opt.zero_grad()
            strategy.outer_opt.step()

            avg = running_loss / strategy.H
            history.append(avg)
            print(f"Rank {self.rank} | outer_step {outer_step:>4d} | avg_loss {avg:.4f}")

        return history
