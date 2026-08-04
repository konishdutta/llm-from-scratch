from cs336_basics.trainer import Optimizer
from cs336_basics.trainer import Trainer
from cs336_basics.trainer import TrainerConfig
from dataclasses import replace
from pathlib import Path

base_config = TrainerConfig(
    run_name = 'base_run',
    device = 'cuda',
    dtype = None,
    checkpoint_dir = Path(__file__).parent.parent.resolve() / "checkpoints",
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
    warmup_iters = 100,
    anneal_iters = 2_000,
    num_iters = 2_000,
    batch_size = 24,
    log_every = 50,
    checkpoint_every = None,
    eval_every = 500,
    eval_batches=10,
    rms_norm=True,
    pre_norm=True,
    seed=69,
)

experiments = [
    replace(
        base_config,
        run_name="no_pos_ablation",
        rope_theta=None,
    ),
    replace(
        base_config,
        run_name="swiglu_ablation",
        d_ff=4 * base_config.d_model,
        ffn_type='silu',
    ),
]

def run_experiments():
    for exp_config in experiments:
        print(
            f"Running {exp_config.run_name}: "
        )
        trainer = Trainer(exp_config)
        trainer.run()

if __name__ == '__main__':
    run_experiments()