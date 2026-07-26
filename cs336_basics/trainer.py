import torch
import torch.nn as nn


def cross_entropy(preds, target):
    """
     −log (exp(o_correct)) + log(sum(exp(o_i))) = log(sum(exp(o_i))) - o_correct
    """
    # preds are [... seq_len vocab_size]
    # target are [... seq_len]
    seq_len = target.shape[-1]
    max_preds, _ = preds.max(dim=-1, keepdim=True)
    scaled_logits = preds - max_preds

    correct = torch.gather(scaled_logits, dim=-1, index=target.unsqueeze(-1))
    log_denom = scaled_logits.exp().sum(dim=-1).log()

    return (log_denom - correct).mean()