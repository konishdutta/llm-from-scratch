from .model import softmax
from .model import TransformerLM
from .tokenizer import BPETokenizer
from .trainer import load_checkpoint
from .trainer import load_config
from pathlib import Path
import torch
from typing import List


def generate_text(
    text,
    model,
    tokenizer,
    end_token,
    max_new_tokens=999,
    temperature=1.0,
    top_p=None,
):
    input_tokens = tokenizer.encode(text)
    end_token_id = tokenizer.reverse_vocab[end_token.encode('utf-8')]
    output_tokens = generate_tokens(model, input_tokens, end_token_id, max_new_tokens, temperature, top_p)
    res = tokenizer.decode(output_tokens)
    return res


def generate_tokens(
    model,
    prompt_tokens,
    end_token_id,
    max_new_tokens = 999,
    temperature = 1.0,
    top_p = None,
) -> List[int]:
    # validate args
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0.")
    if top_p is not None and top_p <= 0.0:
        raise ValueError("top p must be > 0.")
    if top_p is not None and top_p > 1.0:
        raise ValueError("top p must be <= 1.")
    if max_new_tokens <= 0:
        raise ValueError("max new tokens must be > 0.")

    # set up next token run
    next_token = None
    input_tokens = torch.tensor(prompt_tokens, device=model.device, dtype=torch.long).unsqueeze(0) # create bsz of 1
    res = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        while True:
            input_tokens = input_tokens[:, -model.context_length:]
            output_tokens = sample_next_token(model, input_tokens, temperature, top_p)
            next_token = (output_tokens[0, 0]).item()
            if next_token == end_token_id:
                break

            res.append(next_token)

            if len(res) == max_new_tokens:
                break
            # input tokens are [bsz, seq_len]
            # output tokens are [bsz, 1]
            input_tokens = torch.cat((input_tokens, output_tokens), dim=-1)

    model.train(was_training)
    return res



def sample_next_token(model, input_tokens, temperature, top_p):
    logits = model(input_tokens, torch.arange(input_tokens.shape[-1], device=model.device))  # [bsz, seq_len, vocab_size]
    next_token_logits = logits[:, -1, :]
    probs = softmax(next_token_logits/temperature, dim=-1)  # [bsz, vocab_size]

    if top_p is not None:
        probs, indices = torch.sort(probs, dim=-1, descending=True)
        cum_probs = torch.cumsum(probs, dim=-1)
        cum_probs = cum_probs - probs
        mask = (cum_probs < top_p).to(dtype=probs.dtype)
        probs = mask * probs
        probs /= probs.sum(dim=-1, keepdim=True)  # normalize back to probability
        res = torch.multinomial(probs, 1)
        return torch.gather(indices, dim=-1, index=res)

    return torch.multinomial(probs, 1)


def load_model_for_generation(filepath):
    filepath = Path(filepath)
    if not filepath.is_file():
        raise ValueError(f"Not a file: {filepath}")
    
    config = load_config(filepath)
    model = TransformerLM(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        num_layers=config["num_layers"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=config["rope_theta"],
        device=config["device"],
        dtype=config["dtype"],
    )
    load_checkpoint(filepath, model, None)
    return model


if __name__ == '__main__':        
    model = load_model_for_generation(Path(__file__).parent.parent.resolve() / "checkpoints" / "ts_002" / "ts_002_step_80000.pt")
    tokenizer = BPETokenizer.from_param_file(Path(__file__).parent.parent.resolve() / "tokenizer" / "tiny_stories_params.pkl")

    print(f"Welcome to Tiny Stories GPT.")
    print("-" * 60)
    print()

    while True:
        ip = input()

        output = generate_text(ip,
                            model, tokenizer, "<|endoftext|>", max_new_tokens=5000,
                            temperature=0.8, top_p=0.8)
        print(output)
        print("-" * 60)
        print()
