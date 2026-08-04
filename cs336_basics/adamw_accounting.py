
from .resource_accounting import count_flops

def adamw_gpt2_xl_memory(b):
    """
    vocab_size:  50,257
    context_length:  1,024
    num_layers:  48
    d_model:  1,600
    num_heads:  25
    d_ff:  4,288 (the nearest multiple of 64 to 8
    3 × 1, 600)
    """
    rms_floats = 1600 + 1600 * 1024 * b + 1600 + 1600 + 1600
    # simplified
    rms_floats = 1_638_400 * b + 6_400

    attn_floats = 16 * 1600 ** 2 + 3 * 1600 * 1024 * b + 2 * 1024 **2 * 25 * b + 1600 * 1024 * b + 1600 * 1024 * b
    # simplified
    attn_floats = 40_960_000 + 60_620_800 * b

    mlp_floats = 3 * 4288 * 1600 * 4 + 1024 * 4288 * 3 * b + 1600 * 1024 * b
    # simplified
    mlp_floats = 82_329_600 + 14_811_136 * b

    tfmr_floats = mlp_floats + attn_floats + 2 * rms_floats
    tfmr_floats = 123_302_400 + 78_708_736 * b

    output_floats = 4 * 1600 * 50_257
    loss_floats = 2* 1024 * 50257 * b

    total_mem = 4 * (48 * tfmr_floats + output_floats + loss_floats)
    total_mem = 24_960_640_000 + 15_523_782_656 * b
    return total_mem


def adamw_memory(
        bsz,
        vocab_size,
        context_length,
        num_layers,
        d_model,
        num_heads
    ):
    d_ff = round(8 * d_model / (3 * 64)) * 64

    # RMS Norm
    rms_params = d_model
    rms_activations = bsz * d_model * context_length
    rms_grad = d_model
    rms_first_moment = d_model
    rms_second_moment = d_model
    rms_mem = 4 * (rms_params + rms_activations + rms_grad + rms_first_moment + rms_second_moment)
    print(f"Memory usage for 1 RMS-Norm Module: {rms_mem:,} bytes")

    # Multihead attn
    wqkvo_params = 4 * d_model ** 2
    wqkvo_grad = wqkvo_params
    wqkvo_first_moment = wqkvo_params
    wqkvo_second_moment = wqkvo_params
    qkv_activations = 3 * bsz * d_model * context_length
    scores_activations = bsz * num_heads * context_length ** 2
    softmax_activations = bsz * num_heads * context_length ** 2
    softmax_times_values_activations = bsz * d_model * context_length
    attn_activations = bsz * d_model * context_length
    attn_mem = 4 * (wqkvo_params + wqkvo_grad
                    + wqkvo_first_moment + wqkvo_second_moment
                    + qkv_activations + scores_activations
                    + softmax_activations + softmax_times_values_activations
                    + attn_activations)
    print(f"Memory usage for 1 MHA Module: {attn_mem:,} bytes")


    # SwiGLU MLP
    w123_params = 3 * d_ff * d_model
    w123_grad = 3 * d_ff * d_model
    w123_first_moment = w123_grad
    w123_second_moment = w123_grad
    w1_activations = bsz * context_length * d_ff
    w3_activation = bsz * context_length * d_ff
    post_gate_activation = bsz * context_length * d_ff
    w2_activation = bsz * context_length * d_model
    mlp_mem = 4 * (w123_params + w123_grad + w123_first_moment + w123_second_moment
                   + w1_activations + w3_activation + post_gate_activation + w2_activation)
    print(f"Memory usage for 1 SwiGLU MLP Module: {mlp_mem:,} bytes")

    tfmr_mem = 2 * rms_mem + mlp_mem + attn_mem
    print(f"Memory usage for 1 Transformer Module: {tfmr_mem:,} bytes")

    print(f"Memory usage for {num_layers} Transformer Module: {num_layers * tfmr_mem:,} bytes")


    # Output embedding
    output_emb_params = d_model * vocab_size + d_model
    output_emb_grad = output_emb_params
    output_emb_first_moment = output_emb_params
    output_emb_second_moment = output_emb_params    
    output_mem = 4 * (output_emb_params + output_emb_grad + output_emb_first_moment
                      + output_emb_second_moment)
    print(f"Memory usage for Output Module: {output_mem:,} bytes")


    # Cross entropy
    logits_activation = bsz * context_length * vocab_size
    logits_grad = bsz * context_length * vocab_size
    ce_mem = 4 * (logits_activation + logits_grad)
    print(f"Memory usage for Cross Entropy Module: {ce_mem:,} bytes")

    print("-" * 60)
    print(f"TOTAL MEMORY USED: {num_layers * tfmr_mem + output_mem + ce_mem:,} bytes")


def adamw_flops(
        bsz,
        vocab_size,
        context_length,
        num_layers,
        d_model,
        num_heads
    ):
    adamw_multiplier = 1 # weight_decay
    adamw_multiplier += 3 # 2 mult + 1 addition for fm
    adamw_multiplier += 4 # 3 mult + 1 addition for sm
    adamw_multiplier += 5 # 1 sub, 1 add, 1 mult, 1 div, 1 sqrt

    # now count params
    d_ff = round(8 * d_model / (3 * 64)) * 64
    tfmr_params = 4 * d_model **2 # wq, wk, wv, wo
    tfmr_params += 3 * d_ff * d_model # w1, w2, w3
    tfmr_params += 2 * d_model # rms

    tfmr_params *= num_layers
    output_params = vocab_size * d_model
    final_rms_params = d_model
    total_params = tfmr_params + output_params + final_rms_params
    total_flops = adamw_multiplier * total_params
    return total_flops

    


if __name__ == '__main__':
    print("GPT-XL")
    vocab_size = 50_257
    context_length = 1024
    num_layers = 48
    d_model = 1600
    num_heads = 25
    bsz = 1024
    adamw_memory(
        bsz,
        vocab_size,
        context_length,
        num_layers,
        d_model,
        num_heads
    )
    total_adam_flops = adamw_flops(bsz, vocab_size, context_length, num_layers, d_model, num_heads)
    print(f"FLOPS for AdamW Update: {total_adam_flops:,} FLOPs")
    forward_pass_flops = count_flops(vocab_size, context_length, num_layers, d_model)

    total_flops = 3 * bsz * forward_pass_flops + total_adam_flops
    print(f"Total Flops: {total_flops:,} FLOPS")
    H100_flops = 495e12
    num_steps = 400_000
    mfu = 0.5
    flops_by_steps = num_steps * total_flops
    tot_time_sec = flops_by_steps / (H100_flops * mfu)
    tot_time_days = tot_time_sec / (60 * 60 * 24)
    print(f"It'll take {tot_time_days:.2f} days.")