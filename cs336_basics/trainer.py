from .optimizer import AdamWOptimizer
from .optimizer import SGDOptimizer
from .optimizer import get_lr_cosine_schedule
from .model import TransformerLM
from dataclasses import dataclass
from enum import Enum
import numpy as np
from pathlib import Path
import torch
from typing import Tuple
import wandb


class Optimizer(Enum):
    AdamW = AdamWOptimizer
    SGD = SGDOptimizer


@dataclass(kw_only=True)
class TrainerConfig:
    # System settings
    run_name: str
    device: torch.device | str | None = None
    dtype: torch.dtype | None = None
    checkpoint_dir: Path
    snapshot_path: Path | None = None

    # Data settings
    training_path: Path
    validation_path: Path

    # Model hyperparameters
    vocab_size: int
    context_length: int
    num_layers: int
    d_model: int
    num_heads: int
    d_ff: int | None = None
    rope_theta: float | None = None

    # Optimizer hyperparameters
    optimizer: Optimizer = Optimizer.AdamW
    max_lr: float = 1e-3
    min_lr: float = 0
    warmup_iters: int
    anneal_iters: int
    num_iters: int
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-2
    eps: float = 1e-8

    # Trainer hyperparameters
    batch_size: int
    log_every: int | None = 1_000
    checkpoint_every: int | None = None
    eval_every: int | None = None

    def __post_init__(self):
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.training_path = Path(self.training_path)
        self.validation_path = Path(self.validation_path)
        if self.snapshot_path is not None:
            self.snapshot_path = Path(self.snapshot_path)

        if not self.training_path.is_file():
            raise ValueError(f"Training data is not an existing file: {self.training_path}")
        if not self.validation_path.is_file():
            raise ValueError(f"Validation data is not an existing file: {self.validation_path}")
        if self.snapshot_path is not None and not self.snapshot_path.is_file():
            raise ValueError(f"Snapshot file is not an existing file: {self.snapshot_path}")


class Trainer:
    def __init__(self, config: TrainerConfig):
        self.config = config
        self.model = TransformerLM(
            vocab_size=config.vocab_size,
            context_length=config.context_length,
            num_layers=config.num_layers,
            d_model=config.d_model,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            rope_theta=config.rope_theta,
            device=config.device,
            dtype=config.dtype,
        )
        self.optimizer = self._create_optimizer()
        self.train_data = load_memmap(config.training_path)
        self.validation_data = load_memmap(config.validation_path)
        self.starting_step = 0
        if self.config.snapshot_path is not None:
            self.starting_step = load_checkpoint(
                self.config.snapshot_path,
                self.model,
                self.optimizer,
            )
        self.config.checkpoint_dir.mkdir(exist_ok=True, parents=True)


    def _create_optimizer(self):
        if self.config.optimizer.name == 'AdamW':
            return AdamWOptimizer(
                self.model.parameters(),
                lr=self.config.max_lr,
                betas=self.config.betas,
                eps=self.config.eps,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer.name == 'SGD':
            return SGDOptimizer(
                self.model.parameters(),
                lr=self.config.max_lr
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer.name}")


    def run(self):
        with wandb.init(
            project="cs336-assignment1",
            name=self.config.run_name,
            config={
                "learning_rate": self.config.max_lr,
                "batch_size": self.config.batch_size,
                "num_layers": self.config.num_layers,
                "d_model": self.config.d_model,
            },
        ) as run:
            self.model.train()
            for step in range(self.starting_step, self.config.num_iters):
                self.optimizer.zero_grad()

                # set the learning rate
                curr_lr = get_lr_cosine_schedule(
                    step,
                    self.config.max_lr,
                    self.config.min_lr,
                    self.config.warmup_iters,
                    self.config.anneal_iters,
                )
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = curr_lr

                # load data
                sequence, target = load_data(
                    self.train_data,
                    batch_size=self.config.batch_size,
                    context_length=self.config.context_length,
                    device=self.config.device,
                )

                # run the model and backprop
                logits = self.model(sequence)
                loss = cross_entropy(logits, target)
                loss.backward()
                completed_steps = step + 1
                should_log = self._should_log(completed_steps)
                if should_log:
                    grad_norm = self.compute_grad_norm()
                self.optimizer.step()

                # checkpoint and log
                if self._should_checkpoint(completed_steps):
                    out_path = self.config.checkpoint_dir / f"{self.config.run_name}_step_{completed_steps}.pt"
                    save_checkpoint(self.model, self.optimizer, completed_steps, out_path)

                if should_log:
                    self.log_training_metrics(run, completed_steps, loss, grad_norm)
                    

    def log_training_metrics(self, run, completed_steps, loss, grad_norm):
        curr_lr = self.optimizer.param_groups[0]["lr"]
        tokens_seen = self.config.batch_size * self.config.context_length * completed_steps

        run.log(
            {
                "train/loss": loss.item(),
                "train/learning_rate": curr_lr,
                "train/tokens_seen": tokens_seen,
                "train/grad_norm": grad_norm.item(),
            },
            step = completed_steps
        )

    def compute_grad_norm(self):
        sq_grad_norm = torch.tensor(0.0, dtype=torch.float32, device=self.config.device)
        for p in self.model.parameters():
            if p.grad is None:
                continue
            sq_grad_norm += (p.grad.to(dtype=torch.float32) ** 2).sum()
        return sq_grad_norm ** 0.5

    def _should_checkpoint(self, steps):
        if self.config.checkpoint_every is None:
            return False
        return steps % self.config.checkpoint_every == 0 or steps == self.config.num_iters


    def _should_log(self, steps):
        if self.config.log_every is None:
            return False
        return steps % self.config.log_every == 0 or steps == self.config.num_iters


def load_memmap(filepath: str):
    return np.memmap(filepath, dtype=np.uint16, mode='r')


def cross_entropy(logits, target):
    """
     −log (exp(o_correct)) + log(sum(exp(o_i))) = log(sum(exp(o_i))) - o_correct
    """
    # preds are [... seq_len vocab_size]
    # target are [... seq_len]
    max_logits, _ = logits.max(dim=-1, keepdim=True)
    scaled_logits = logits - max_logits

    correct = torch.gather(scaled_logits, dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
    log_denom = scaled_logits.exp().sum(dim=-1).log()

    return (log_denom - correct).mean()


def load_data(
        x: np.ndarray,
        batch_size: int,
        context_length: int,
        device: torch.device | str) -> Tuple[torch.Tensor, torch.Tensor]:
    valid_starting_positions = len(x) - context_length
    if valid_starting_positions <= 0:
        raise ValueError("Context length too large for data.")
    
    start_idx = np.random.randint(low=0, high=valid_starting_positions, size=(batch_size,1))
    seq_offsets = np.expand_dims(np.arange(context_length), axis=0)
    seq_idx = start_idx + seq_offsets
    sequence = torch.from_numpy(x[seq_idx]).to(device=device, dtype=torch.long)
    target = torch.from_numpy(x[1 + seq_idx]).to(device=device, dtype=torch.long)

    return sequence, target


def save_checkpoint(model, optimizer, iteration, out):
    obj = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration}
    torch.save(obj, out)


def load_checkpoint(src, model, optimizer):
    custom_class = (None, "tests.test_serialization._TestNet")
    with torch.serialization.safe_globals([getattr, custom_class]):
        obj = torch.load(src)
    model_checkpoint, optimizer_checkpoint = obj.get("model", None), obj.get("optimizer", None)
    if model_checkpoint is not None:
        model.load_state_dict(model_checkpoint)
    if optimizer_checkpoint is not None:
        optimizer.load_state_dict(optimizer_checkpoint)
    return obj.get("iteration", 0)


if __name__ == '__main__':
    config = TrainerConfig(
        run_name = "smoke_test_001",
        device = "cuda",
        dtype = torch.float32,
        checkpoint_dir = Path(__file__).parent.parent.resolve() / "checkpoints" / "smoke_test",
        snapshot_path = None,
        training_path = Path(__file__).parent.parent.resolve() / "tokenized_data" / "ts_val.bin",
        validation_path = Path(__file__).parent.parent.resolve() / "tokenized_data" / "ts_val.bin",
        vocab_size = 10_000,
        context_length = 64,
        num_layers = 6,
        d_model = 256,
        num_heads = 8,
        d_ff = None,
        rope_theta = 10_000.0,
        optimizer = Optimizer.AdamW,
        max_lr = 1e-3,
        min_lr = 0,
        warmup_iters = 2,
        anneal_iters = 5,
        num_iters = 5,
        batch_size = 4,
        log_every = 1,
        checkpoint_every = 2,
    )

    trainer = Trainer(config)
    trainer.run()