import torch
import torch.nn as nn


def cross_entropy(preds, target):
    """
     −log (exp(o_correct)) + log(sum(exp(o_i))) = log(sum(exp(o_i))) - o_correct
    """
    # preds are [... seq_len vocab_size]
    # targets are [... seq_len]
    seq_len = target.shape[-1]
    max_preds, _ = preds.max(dim=-1, keepdim=True)
    scaled_logits = preds - max_preds

    correct = scaled_logits[..., torch.arange(seq_len), target[..., torch.arange(seq_len)]]
    denom = scaled_logits.exp().sum(dim=-1).log()

    return (denom - correct).mean()