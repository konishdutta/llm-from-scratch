import regex as re
from collections import defaultdict


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

text = """low low low low low
lower lower widest widest widest
newest newest newest newest newest newest"""

pretokens = re.findall(PAT, text)

# Initialize the vocabulary
encoder = {}
encoder["<|endoftext|>".encode('utf-8')] = 0
for i in range(256):
    encoder[bytes([i])] = i + 1

# --- MERGE LOGIC ---
def apply_merge(encoder, pretoken_to_symbols, pretoken_counts) -> None:
    """
    One round of the merging. Modifies encoder and maps in-place
    """
    pair_counts = defaultdict(int)
    for word, token_list in pretoken_to_symbols.items():
        for i in range(1, len(token_list)):
            pair_counts[(token_list[i-1], token_list[i])] += pretoken_counts[word]
    next_merge = max(pair_counts.items(), key=lambda x: (x[1], x[0]))[0]
    next_merge_bytes = b"".join(next_merge)
    encoder[next_merge_bytes] = len(encoder)
    for word, tokens in pretoken_to_symbols.items():
        res = []
        i = 0
        while i < len(tokens):
            if i < len(tokens)-1 and tokens[i] == next_merge[0] and tokens[i+1] == next_merge[1]:
                res.append(next_merge_bytes)
                i += 2
            else:
                res.append(tokens[i])
                i += 1
        pretoken_to_symbols[word] = res

# Seed the initial word2token map
pretoken_bytes = [p.encode('utf-8') for p in pretokens]
pretokens_set = set(pretoken_bytes)
pretoken_counts = {}
for pretoken in pretokens_set:
    pretoken_counts[pretoken] = pretoken_bytes.count(pretoken)
pretoken_to_symbols = defaultdict(list)
for t in pretokens_set:
    for c in t:
        pretoken_to_symbols[t].append(bytes([c]))

for _ in range(6):
    apply_merge(encoder, pretoken_to_symbols, pretoken_counts)
    print("new token:", list(encoder.keys())[-1])