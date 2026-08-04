from .optimizer import AdamWOptimizer
from .optimizer import AuroraOptimizer
from .optimizer import SGDOptimizer
from .optimizer import get_lr_cosine_schedule
from .model import TransformerLM
from dataclasses import dataclass
from enum import Enum
import numpy as np
from pathlib import Path
import random
import time
import torch
from typing import Tuple
import wandb


class Optimizer(Enum):
    AdamW = AdamWOptimizer
    SGD = SGDOptimizer
    Aurora = AuroraOptimizer


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
    rms_norm: bool = True
    pre_norm: bool = True
    ffn_type: str = 'swiglu'

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
    aurora_mu: float = 0.95
    aurora_beta: float = 0.5
    max_aurora_lr: float = 1e-3
    min_aurora_lr: float = 0

    # Trainer hyperparameters
    batch_size: int
    log_every: int | None = 1_000
    checkpoint_every: int | None = None
    eval_every: int | None = None
    eval_batches: int = 20
    seed: int | None = None

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
        if config.seed is not None:
            random.seed(config.seed)
            np.random.seed(config.seed)
            torch.manual_seed(config.seed)
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
            rms_norm=config.rms_norm,
            pre_norm=config.pre_norm,
            ffn_type=config.ffn_type,
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
        if self.config.rope_theta is not None:
            self.token_positions = torch.arange(self.config.context_length, device=self.config.device)
        else:
            self.token_positions = None


    def _create_optimizer(self):
        if self.config.optimizer.name == 'AdamW':
            return {"adamw": AdamWOptimizer(
                self.model.parameters(),
                lr=self.config.max_lr,
                betas=self.config.betas,
                eps=self.config.eps,
                weight_decay=self.config.weight_decay,
            )}
        elif self.config.optimizer.name == 'SGD':
            return {"sgd": SGDOptimizer(
                self.model.parameters(),
                lr=self.config.max_lr
            )}
        elif self.config.optimizer.name == 'Aurora':
            print("Generating Aurora Optimizer")
            print(f"{'optimizer':<10} | {'parameter name':<45} | {'shape':<20}")
            print("-" * (10 + 20 + 45 + 6))
            # Use AdamW for non-matrix weights
            adam_names = {
                'token_embeddings',
                'ln1',
                'ln2',
                'ln_final',
                'lm_head',
            }
            adam_params = []
            aurora_params = []
            for name, param in self.model.named_parameters():
                if any(keyword in name for keyword in adam_names) or param.ndim < 2:
                    adam_params.append(param)
                    print(f"{'AdamW':<10} | {name:<45} | {str(tuple(param.shape)):<20}")
                else:
                    aurora_params.append(param)
                    print(f"{'Aurora':<10} | {name:<45} | {str(tuple(param.shape)):<20}")
            aurora = AuroraOptimizer(
                aurora_params,
                lr=self.config.max_aurora_lr,
                weight_decay=self.config.weight_decay,
                mu=self.config.aurora_mu,
                beta=self.config.aurora_beta,
                eps=self.config.eps
            )
            adam = AdamWOptimizer(
                adam_params,
                lr=self.config.max_lr,
                betas=self.config.betas,
                eps=self.config.eps,
                weight_decay=self.config.weight_decay,
            )
            
            return {"adamw": adam, "aurora": aurora}

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
                "num_iters": self.config.num_iters,
                "seed": self.config.seed,
                "rms_norm": self.config.rms_norm,
                "pre_norm": self.config.pre_norm,
            },
        ) as run:
            run_start_time = time.perf_counter()
            last_log_time = run_start_time
            last_log_step = self.starting_step
            self.model.train()
            for step in range(self.starting_step, self.config.num_iters):
                metrics = {}
                for name, optimizer in self.optimizer.items():
                    optimizer.zero_grad()

                # set the learning rate
                curr_lr = get_lr_cosine_schedule(
                    step,
                    self.config.max_lr,
                    self.config.min_lr,
                    self.config.warmup_iters,
                    self.config.anneal_iters,
                )

                # optional if using Aurora
                curr_aurora_lr = get_lr_cosine_schedule(
                    step,
                    self.config.max_aurora_lr,
                    self.config.min_aurora_lr,
                    self.config.warmup_iters,
                    self.config.anneal_iters,
                )

                for name, optimizer in self.optimizer.items():
                    for param_group in optimizer.param_groups:
                        if name == 'aurora':
                            param_group['lr'] = curr_aurora_lr
                        else:
                            param_group['lr'] = curr_lr


                # load data
                sequence, target = load_data(
                    self.train_data,
                    batch_size=self.config.batch_size,
                    context_length=self.config.context_length,
                    device=self.config.device,
                )

                # run the model and backprop
                logits = self.model(sequence, token_positions=self.token_positions)
                loss = cross_entropy(logits, target)
                loss.backward()
                completed_steps = step + 1
                should_log = self._should_log(completed_steps)
                if should_log:
                    grad_norm = self.compute_grad_norm()

                for name, optimizer in self.optimizer.items():
                    optimizer.step()

                # checkpoint and log
                if self._should_checkpoint(completed_steps):
                    out_path = self.config.checkpoint_dir / f"{self.config.run_name}_step_{completed_steps}.pt"
                    save_checkpoint(self.config, self.model, self.optimizer, completed_steps, out_path)

                if should_log:
                    metrics |= self.build_training_metrics(loss, grad_norm, completed_steps)

                if self._should_eval(completed_steps):
                    loss = self.run_eval()
                    metrics |= self.build_eval_metrics(loss)

                if metrics:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    now = time.perf_counter()
                    elapsed_seconds = now - run_start_time
                    interval_seconds = now - last_log_time
                    interval_steps = completed_steps - last_log_step
                    interval_tokens = (
                        interval_steps
                        * self.config.batch_size
                        * self.config.context_length
                    )
                    metrics |= {
                        "perf/session_wall_clock_seconds": elapsed_seconds,
                        "perf/seconds_per_step": interval_seconds / interval_steps,
                        "perf/tokens_per_second": interval_tokens / interval_seconds,
                    }

                    run.log(metrics, step=completed_steps)
                    last_log_time = now
                    last_log_step = completed_steps


    def build_training_metrics(self, loss, grad_norm, completed_steps):
        tokens_seen = self.config.batch_size * self.config.context_length * completed_steps
        metrics = {
                "train/loss": loss.item(),
                "train/tokens_seen": tokens_seen,
                "train/grad_norm": grad_norm.item(),
            }
        if "aurora" in self.optimizer:
            metrics |= {"train/lr_aurora": self.optimizer['aurora'].param_groups[0]['lr']}
        if "adamw" in self.optimizer:
            metrics |= {"train/lr_adamw": self.optimizer['adamw'].param_groups[0]['lr']}
        if "sgd" in self.optimizer:
            metrics |= {"train/lr_sgd": self.optimizer['sgd'].param_groups[0]['lr']}

        return metrics


    def run_eval(self):
        self.model.eval()
        loss = torch.tensor(0.0, dtype=torch.float32, device=self.config.device)
        with torch.no_grad():
            for _ in range(self.config.eval_batches):
                sequence, target = load_data(
                    self.validation_data,
                    batch_size=self.config.batch_size,
                    context_length=self.config.context_length,
                    device=self.config.device,
                )
                logits = self.model(sequence, token_positions=self.token_positions)
                loss += cross_entropy(logits, target)

            loss /= self.config.eval_batches
        self.model.train()
        return loss
        

    def build_eval_metrics(self, loss):
        metrics = {
                "eval/loss": loss.item(),
                "eval/perplexity": torch.exp(loss).item()
            }
        return metrics


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

    def _should_eval(self, steps):
        if self.config.eval_every is None:
            return False
        return steps % self.config.eval_every == 0 or steps == self.config.num_iters


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


def save_checkpoint(config, model, optimizer, iteration, out):
    model_config = {
        "vocab_size": config.vocab_size,
        "context_length": config.context_length,
        "num_layers": config.num_layers,
        "d_model": config.d_model,
        "num_heads": config.num_heads,
        "d_ff": config.d_ff,
        "rope_theta": config.rope_theta,
        "device": config.device,
        "dtype": config.dtype,
        "rms_norm": config.rms_norm,
        "pre_norm": config.pre_norm,
        "ffn_type": config.ffn_type,
    }

    optimizer_state = {name: opt.state_dict() for name, opt in optimizer.items()}

    obj = {"model": model.state_dict(), "optimizer": optimizer_state, "iteration": iteration, "config": model_config}
    torch.save(obj, out)


def load_checkpoint(src, model, optimizer):
    custom_class = (None, "tests.test_serialization._TestNet")
    with torch.serialization.safe_globals([getattr, custom_class]):
        obj = torch.load(src)
    model_checkpoint, optimizer_checkpoint = obj.get("model", None), obj.get("optimizer", None)
    if model_checkpoint is not None and model is not None:
        model.load_state_dict(model_checkpoint)
    if optimizer_checkpoint is not None and optimizer is not None:
        for name, opt_state in optimizer_checkpoint.items():
            if name not in optimizer:
                raise ValueError(f"got {name} in saved checkpoint, but not in config.")
            optimizer[name].load_state_dict(opt_state)
    return obj.get("iteration", 0)


def load_config(src):
    custom_class = (None, "tests.test_serialization._TestNet")
    with torch.serialization.safe_globals([getattr, custom_class]):
        obj = torch.load(src)

    return obj["config"]


def smoke_test():
    config = TrainerConfig(
        run_name = "smoke_test_aurora_001",
        device = "cuda",
        dtype = torch.float32,
        checkpoint_dir = Path(__file__).parent.parent.resolve() / "checkpoints" / "smoke_test_aurora",
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
        optimizer = Optimizer.Aurora,
        max_lr = 1e-3,
        min_lr = 0,
        warmup_iters = 2,
        anneal_iters = 5,
        num_iters = 5,
        batch_size = 4,
        log_every = 1,
        checkpoint_every = 2,
        eval_every=1,
        eval_batches=2,
    )

    trainer = Trainer(config)
    trainer.run()


def resume_smoke():
    config = TrainerConfig(
        run_name = "resume_smoke_aurora_001",
        device = "cuda",
        dtype = torch.float32,
        checkpoint_dir = Path(__file__).parent.parent.resolve() / "checkpoints" / "resume_smoke_aurora",
        snapshot_path = Path(__file__).parent.parent.resolve() / "checkpoints" / "smoke_test_aurora" / "smoke_test_aurora_001_step_2.pt",
        training_path = Path(__file__).parent.parent.resolve() / "tokenized_data" / "ts_val.bin",
        validation_path = Path(__file__).parent.parent.resolve() / "tokenized_data" / "ts_val.bin",
        vocab_size = 10_000,
        context_length = 64,
        num_layers = 6,
        d_model = 256,
        num_heads = 8,
        d_ff = None,
        rope_theta = 10_000.0,
        optimizer = Optimizer.Aurora,
        max_lr = 1e-3,
        min_lr = 0,
        warmup_iters = 2,
        anneal_iters = 5,
        num_iters = 5,
        batch_size = 4,
        log_every = 1,
        checkpoint_every = 2,
        eval_every=1
    )

    trainer = Trainer(config)
    trainer.run()


def train_tiny_stories():
    config = TrainerConfig(
        run_name = "ts_002",
        device = "cuda",
        dtype = torch.float32,
        checkpoint_dir = Path(__file__).parent.parent.resolve() / "checkpoints" / "ts_002",
        snapshot_path = None,
        training_path = Path(__file__).parent.parent.resolve() / "tokenized_data" / "ts_train.bin",
        validation_path = Path(__file__).parent.parent.resolve() / "tokenized_data" / "ts_val.bin",
        vocab_size = 10_000,
        context_length = 256,
        num_layers = 4,
        d_model = 512,
        num_heads = 16,
        d_ff = 1344,
        rope_theta = 10_000.0,
        optimizer = Optimizer.AdamW,
        max_lr = 1e-3,
        min_lr = 0,
        warmup_iters = 8_000,
        anneal_iters = 80_000,
        num_iters = 80_000,
        batch_size = 16,
        log_every = 50,
        checkpoint_every = 10_000,
        eval_every = 250,
        eval_batches=10,
    )

    trainer = Trainer(config)
    trainer.run()

if __name__ == '__main__':
    smoke_test()
    resume_smoke()