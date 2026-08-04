def count_flops(vocab_size, context_length, num_layers, d_model):
    d_ff = 64 * round(d_model * 8/(3 * 64))
    attn_flops = 8 * context_length * d_model ** 2 + 4 * context_length **2 * d_model
    print(f"Each attention layer costs \t{attn_flops:,} FLOPS.")
    attn_flops *= num_layers
    print(f"All attention layers costs \t{attn_flops:,} FLOPS.")

    ffn_flops = 6 * d_ff * d_model * context_length
    print(f"Each FFN layer costs \t\t{ffn_flops:,} FLOPS.")
    ffn_flops *= num_layers
    print(f"All FFN layers costs \t\t{ffn_flops:,} FLOPS.")

    head_flops = 2 * context_length * d_model * vocab_size
    print(f"Final LM head costs \t\t{head_flops:,} FLOPS.")

    total_flops = attn_flops + ffn_flops + head_flops
    print()
    print(f"Total flops used: \t\t{total_flops:,}")
    print(f"Attn ratio: {attn_flops / total_flops: .2%}")
    print(f"FFN ratio: {ffn_flops / total_flops: .2%}")
    print(f"Head ratio: {head_flops / total_flops: .2%}")

    print("-" * 60)
    return total_flops

if __name__ == '__main__':
    vocab_size = 50_257
    context_length = 1024
    # GPT2 XL
    num_layers = 48
    d_model = 1600
    print("GPT-2 XL")
    count_flops(vocab_size, context_length, num_layers, d_model)

    # GPT2 L
    num_layers = 36
    d_model = 1280
    print("GPT-2 L")
    count_flops(vocab_size, context_length, num_layers, d_model)

    # GPT2 M
    num_layers = 24
    d_model = 1024
    print("GPT-2 M")
    count_flops(vocab_size, context_length, num_layers, d_model)

    # GPT2 S
    num_layers = 12
    d_model = 768
    print("GPT-2 S")
    count_flops(vocab_size, context_length, num_layers, d_model)

    # GPT2 XL with long context
    num_layers = 48
    d_model = 1600
    context_length = 16_384
    print("GPT-2 XL, context length of 16,384")
    count_flops(vocab_size, context_length, num_layers, d_model)

"""
As we get smaller models (lower d_model, num_layers), we move from
FFN being the largest to spreading more evenly across attention and head

But with long context, Attn takes up most of the FLOPs. We also use an
order of magnitude higher of flops.
"""