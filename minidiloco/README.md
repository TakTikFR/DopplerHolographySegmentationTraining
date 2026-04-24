# MiniDiLoCo

Minimal PyTorch implementation of **DiLoCo** (Distributed Low-Communication training).

## Project structure

```
minidiloco/
├── aggregator.py   # AllReduce gradient aggregation
├── data.py         # Wikitext-2 dataloader
├── model.py        # GPT-2 model factory
├── strategy.py     # DiLoCo strategy + optimizer factories
├── train.py        # Distributed Trainer
├── utils.py        # setup / cleanup helpers
├── benchmark.ipynb # Optimizer benchmark (single-GPU)
└── test.ipynb      # Distributed test (multi-GPU)
```

## Pluggable optimizers

`strategy.py` exposes four ready-made optimizer factories:

| Factory | Use for |
|---|---|
| `make_adamw(lr, betas, eps, weight_decay)` | inner loop (default) |
| `make_adam(lr, betas, eps)` | inner loop alternative |
| `make_sgd(lr, momentum, nesterov)` | outer loop (default) |
| `make_rmsprop(lr, alpha, eps)` | outer loop alternative |

```python
from strategy import Diloco, make_adam, make_sgd

strategy = Diloco(
    inner_optimizer_factory=make_adam(lr=1e-3),
    outer_optimizer_factory=make_sgd(lr=0.7),
    H=500,
)
```

## Multi-GPU launch (torchrun)

```bash
torchrun --nproc_per_node=4 -m train   # adapt as needed
```

## Benchmark

Open `benchmark.ipynb` to run the optimizer grid-search and visualise results.
