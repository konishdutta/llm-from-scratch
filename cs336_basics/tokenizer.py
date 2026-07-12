import os
from pathlib import Path
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
import regex as re
from typing import BinaryIO

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


@dataclass(frozen=True)
class BPETokenizerParams:
    vocab: dict[int, bytes]     # index -> bytes
    merges: dict[tuple[int, int], int]  # index1,index2 -> new_index


class BPETokenizer:
    def __init__(self, params: BPETokenizerParams):
        self.params = params


    def encode(self, string: str) -> list[int]:
        indices = [int(s) for s in string.encode('utf-8')]
        try_merge = True
        while try_merge:
            try_merge = False
            possible_merges = []
            for i, j in zip(indices, indices[1:]):
                possible_merge = self.params.merges.get((i, j))
                if possible_merge:
                    possible_merges.append(((i, j), possible_merge))
            if possible_merges:
                try_merge = True
                next_merge = min(possible_merges, key=lambda x: x[1])
                indices = _merge(indices, next_merge[0], next_merge[1])
        return indices


    def decode(self, indices: list[int]) -> str:
        bytes_list = [self.params.vocab.get(b) for b in indices]
        string = b"".join(bytes_list).decode('utf-8')
        return string


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def get_compression_ratio(string: str, indices: list[int]) -> float:
    num_bytes = len(bytes(string, encoding='utf-8'))
    num_tokens = len(indices)
    return num_bytes / num_tokens


def _count_adjacent(
        pretoken_counts: dict[tuple[int, ...], int]
        ) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], set[tuple[int, ...]]]]:
    res = defaultdict(int)
    pair_to_pretokens = defaultdict(set)
    for idx, val in pretoken_counts.items():
        for i, j in zip(idx, idx[1:]):
            res[(i, j)] += val
            pair_to_pretokens[(i, j)].add(idx)
    return res, pair_to_pretokens


def _count_pairs(indices: Sequence[int]) -> dict[tuple[int, int], int]:
    counts = defaultdict(int)
    for i, j in zip(indices, indices[1:]):
        counts[(i, j)] += 1
    return counts


def _merge(
        indices: Sequence[int],
        pair: tuple[int, int],
        new_index: int
        ) -> list[int]:
    new_indices = []
    i = 0
    while i < len(indices):
        if i + 1 < len(indices) and (indices[i], indices[i+1]) == pair:
            new_indices.append(new_index)
            i += 2
        else:
            new_indices.append(indices[i])
            i += 1

    return new_indices

def _pretokenize(
        input_path: str | os.PathLike,
        special_tokens: list[str],
        ) -> dict[tuple[int, ...], int]:
    with open(input_path, "r", encoding="utf-8") as file:
        text = file.read()
    
    # split by special tokens
    if special_tokens:
        splitters = "|".join(map(re.escape, special_tokens))
        text_sections = re.split(splitters, text)
    else:
        text_sections = [text]

    # populate pretokens
    pretokens = []
    for section in text_sections:
        for match in re.finditer(PAT, section):
            pretoken = match.group()
            pretokens.append(pretoken.encode('utf-8'))

    # byte for byte list
    indices = [list(pretoken) for pretoken in pretokens]
    pretoken_counts = defaultdict(int)
    for pretoken in indices:
        pretoken_counts[tuple(pretoken)] += 1
    
    return pretoken_counts


def _pretokenize_chunk(
        input_path: str | os.PathLike,
        special_tokens: list[str],
        start: int,
        end: int,
    ) -> dict[tuple[int, ...], int]:
    with open(input_path, "rb") as file:
        file.seek(start)
        chunk_bytes = file.read(end - start)
    
    text = chunk_bytes.decode(encoding='utf-8')
    # split by special tokens
    if special_tokens:
        splitters = "|".join(map(re.escape, special_tokens))
        text_sections = re.split(splitters, text)
    else:
        text_sections = [text]

    # populate pretokens
    pretokens = []
    for section in text_sections:
        for match in re.finditer(PAT, section):
            pretoken = match.group()
            pretokens.append(pretoken.encode('utf-8'))

    # byte for byte list
    indices = [list(pretoken) for pretoken in pretokens]
    pretoken_counts = defaultdict(int)
    for pretoken in indices:
        pretoken_counts[tuple(pretoken)] += 1
    
    return pretoken_counts

def run_train_bpe(
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str]
        )-> BPETokenizerParams:
    
    num_processes = os.cpu_count()

    pretoken_counts = _pretokenize(input_path, special_tokens)

    merges: dict[tuple[int, int], int] = {}    # (idx1, idx2) -> new_idx
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}
    for i, t in enumerate(special_tokens):
        vocab[i+256] = t.encode('utf-8')
    
    base_vocab_len = len(special_tokens) + 256
    if vocab_size < base_vocab_len:
        raise ValueError("desired vocab size too small.")
    num_merges = vocab_size - base_vocab_len
    counts, pair_to_pretokens = _count_adjacent(pretoken_counts)

    for i in range(num_merges):
        # find the next merge
        if not counts:
            break
        pair = max(counts, key=lambda x: (counts[x], vocab[x[0]], vocab[x[1]]))
        # populate the fields
        new_index = base_vocab_len + i
        merges[pair] = new_index
        vocab[new_index] = vocab[pair[0]] + vocab[pair[1]]

        # update pretoken counts
        pretokens_to_visit = set(pair_to_pretokens[pair])
        pretoken_merge_dict = {p:pretoken_counts[p] for p in pretokens_to_visit}
        for pretoken, val in pretoken_merge_dict.items():
            old_pretoken = pretoken
            old_counts = _count_pairs(pretoken)
            pretoken = _merge(pretoken, pair, new_index)
            new_counts = _count_pairs(pretoken)

            # update counts and pair to pretokens
            delta_keys = old_counts.keys() | new_counts.keys()
            for d in delta_keys:
                new = new_counts.get(d, 0)
                old = old_counts.get(d, 0)
                delta = new - old
                counts[d] += (delta * val)
                if counts[d] == 0:
                    del counts[d]
                if old > 0:
                    pair_to_pretokens[d].remove(old_pretoken)
                if new > 0:
                    pair_to_pretokens[d].add(tuple(pretoken))
            del pretoken_counts[old_pretoken]
            pretoken_counts[tuple(pretoken)] += val

    return BPETokenizerParams(vocab=vocab, merges=merges)



if __name__ == '__main__':
    path = Path(__file__).parent / "test_files" / "test1.txt"
    text =  "Hello, 🌍! 你好!"
    params = run_train_bpe(path, 261, [])
    tokenizer = BPETokenizer(params)
    e = tokenizer.encode(text)
    print(e)
    d = tokenizer.decode(e)
    print(d)
    assert d == text