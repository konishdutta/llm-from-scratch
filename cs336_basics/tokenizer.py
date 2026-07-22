import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
from pathlib import Path
from collections import defaultdict
from collections.abc import Sequence
import regex as re
from typing import BinaryIO, Iterable, Iterator
import multiprocessing
from collections import Counter
import pickle
import heapq


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class BPETokenizerParams:
    def __init__(
        self,
        vocab: dict[int, bytes],     # index -> bytes
        merges: dict[tuple[int, int], int],  # index1,index2 -> new_index
    ) -> None:
        self.vocab = vocab
        self.merges = merges
    

    @classmethod
    def from_vocab_merge_files(
        cls,
        vocab_path: str | os.PathLike,
        merges_path: str | os.PathLike,
    ) -> "BPETokenizerParams":
        with open(vocab_path, 'rb') as vocab_file:
            vocab = pickle.load(vocab_file)
        with open(merges_path, 'rb') as merges_file:
            merges = pickle.load(merges_file)
        return cls.from_vocab_and_ordered_merges(vocab, merges)
    

    @classmethod
    def from_vocab_and_ordered_merges(
        cls,
        vocab: dict[int, bytes],
        ordered_merges_list: list[tuple[bytes, bytes]]
    ) -> "BPETokenizerParams":
        # Assumption: The token ID preserves merge rank
        reverse_vocab = {v: k for k, v in vocab.items()}
        merges = {(reverse_vocab[left_byte], reverse_vocab[right_byte]): reverse_vocab[_concat_bytes(left_byte, right_byte)]
                  for left_byte, right_byte in ordered_merges_list}
        return cls(vocab, merges)    


class BPETokenizer:
    def __init__(self, params: BPETokenizerParams, special_tokens: list[str] | None=None):
        self.params = params
        self.reverse_vocab = {v: k for k, v in self.params.vocab.items()} # token_bytes -> token_id
        self.special_tokens = set(special_tokens or [])
        self.ordered_special_tokens = sorted(self.special_tokens, key=lambda x: -len(x))
        try:
            self.special_tokens_to_id = {token: self.reverse_vocab[token.encode('utf-8')]
                                         for token in self.special_tokens}
        except KeyError as error:
            raise ValueError(
                    f"Special token is missing from vocabulary: {error.args[0]!r}"
                ) from error


    @classmethod
    def from_vocab_merge_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ) -> "BPETokenizer":
        params = BPETokenizerParams.from_vocab_merge_files(vocab_filepath, merges_filepath)
        return cls(params, special_tokens)
    

    @classmethod
    def from_param_file(
        cls,
        param_path: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ) -> "BPETokenizer":
        with open(param_path, 'rb') as file:
            params = pickle.load(file)
        if not isinstance(params, BPETokenizerParams):
            raise ValueError(f"Expected BPETokenizerParams, got {type(params).__name__}")
        return cls(params, special_tokens)
    

    def encode(self, string: str) -> list[int]:
        # 1. Split into sections based on special tokens
        if self.special_tokens:
            pattern = f"({'|'.join(re.escape(token) for token in self.ordered_special_tokens)})"
            sections = re.split(pattern, string)
        else:
            sections = [string]
        res = []
        
        # 2. Encode each section independently
        for section in sections:
            # 2a. Emit special tokens directly without BPE encoding
            if self.special_tokens and section in self.special_tokens:
                res.append(self.special_tokens_to_id[section])
                continue
            
            # 2b. Pretokenize regular text and apply BPE merges
            for match in re.finditer(PAT, section):
                pretoken = self._encode_pretoken(match.group())
                res.extend(pretoken)
        
        return res
    
    
    def encode_iterable(self, iterable: Iterable[str],) -> Iterator[int]:
        for string in iterable:
            if string in self.special_tokens:
                yield self.special_tokens_to_id[string]
            else:
                yield from self.encode(string)
    

    def _encode_pretoken(self, text: str) -> list[int]:
        pretoken = [self.reverse_vocab[bytes([x])] for x in text.encode('utf-8')]

        # Greedily apply highest priority merges
        while True:
            best_pair, best_merged_token_id = None, float('inf')
            for left_token, right_token in zip(pretoken, pretoken[1:]):
                candidate_merge_id = self.params.merges.get((left_token, right_token), float('inf'))
                if candidate_merge_id < best_merged_token_id:
                    best_pair, best_merged_token_id = (left_token, right_token), candidate_merge_id

            if best_pair is None:
                break

            pretoken = _merge(pretoken, best_pair, best_merged_token_id)
        
        return pretoken
        

    def decode(self, indices: list[int]) -> str:
        bytes_list = [self.params.vocab.get(b) for b in indices]
        string = b"".join(bytes_list)
        return string.decode('utf-8', errors='replace')


def _concat_bytes(byte1, byte2):
    return _to_byte(byte1) + _to_byte(byte2)


def _to_byte(b):
    return bytes([b]) if isinstance(b, int) else bytes(b)


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

def _pretokenize_chunk_from_task(
        task: tuple[str | os.PathLike, list[str], int, int]
    ) -> dict[tuple[int, ...], int]:
    return _pretokenize_chunk(*task)


def _find_special_suffix(text: str, special_tokens: list[str]) -> int:
    """
    Returns starting index of possible special token. If no token exists
    return len(text)
    """
    res = len(text)
    for t in special_tokens:
        for i in range(1, len(t)):
            if text.endswith(t[:i]):
                res = min(res, len(text) - i)
    return res


def _pretokenize_chunk(
        input_path: str | os.PathLike,
        special_tokens: list[str],
        start: int,
        end: int,
        chunk_size = 32 * 1024 * 1024,
    ) -> dict[tuple[int, ...], int]:
    if special_tokens:
        ordered_special_tokens = sorted(set(special_tokens), key=len, reverse=True)
        pattern = "|".join(re.escape(token) for token in ordered_special_tokens)
    else:
        pattern = None
    pretoken_counts = defaultdict(int)
    with open(input_path, "rb") as file:
        file.seek(start)
        curr = start
        leftover = b""
        while curr < end:
            nxt = min(end, curr + chunk_size)
            chunk_bytes = file.read(nxt - curr)
            chunk_bytes = leftover + chunk_bytes
            leftover = b""
            try:
                text = chunk_bytes.decode(encoding='utf-8')
            except UnicodeDecodeError as e:
                text = (chunk_bytes[:e.start]).decode('utf-8')
                leftover += chunk_bytes[e.start:]
            del chunk_bytes

            if nxt != end:
                safe_end = _find_special_suffix(text, special_tokens)
                leftover = text[safe_end:].encode('utf-8') + leftover
                text = text[:safe_end]
            if pattern:
                sections = re.split(pattern, text)
            else:
                sections = [text]
            for i, section in enumerate(sections):
                pending_pretoken = None
                for match in re.finditer(PAT, section):
                    last_pretoken, pending_pretoken = pending_pretoken, (match.group()).encode('utf-8')
                    if last_pretoken:
                        pretoken_counts[tuple(last_pretoken)] += 1
                if i < len(sections)-1 and pending_pretoken:
                    pretoken_counts[tuple(pending_pretoken)] += 1

            if nxt != end and pending_pretoken:
                leftover = pending_pretoken + leftover
            elif nxt == end and pending_pretoken:
                pretoken_counts[tuple(pending_pretoken)] += 1

            curr = nxt

    return pretoken_counts

def _heappush(heap, pair, count, tiebreak):
    heapq.heappush(heap, (-count, tiebreak[pair[0]], tiebreak[pair[1]], pair))


def _count_pretokens_parallel(input_path: str | os.PathLike, special_tokens: list[str]) -> Counter[tuple[int, ...]]:
    """
    Count byte-level pretokens across the corpus using multiprocessing.

    Special tokens act as boundaries so ordinary pretokens are never joined
    across them.

    Returns a Counter whose keys are pretokens represented as tuples of token IDs
    and whose values are corpus frequencies. Initially, each token ID is the
    integer value of one byte; later BPE merges may replace adjacent IDs with
    newly assigned token IDs.
    """
    num_processes = os.cpu_count()
    # Datasets we intend to use allow us to hardcode special token.
    with open(input_path, "rb") as file:
        boundaries = find_chunk_boundaries(file, num_processes, b"<|endoftext|>")
    tasks = [(input_path, special_tokens, start, end) for start, end in zip(boundaries, boundaries[1:])]

    pretoken_counts = Counter()
    with multiprocessing.Pool() as pool:
        for chunk_counts in pool.imap_unordered(_pretokenize_chunk_from_task, tasks):
            pretoken_counts.update(chunk_counts)
    
    return pretoken_counts


def _find_best_pair(heap, counts, tiebreak):
    """
    Given a heap, return the best pair of tokens to merge
    """
    # Lazy invalidation leaves stale entries in the heap, so periodically
    # rebuild it from the authoritative `counts` mapping.
    if len(heap) > len(counts) * 4:
        heap = []
        for pair, count in counts.items():
            _heappush(heap, pair, count, tiebreak)

    # Discard candidates whose saved count no longer matches `counts`.
    neg_count, _, _, pair = heapq.heappop(heap)
    while -1 * neg_count != counts.get(pair, 0):
        neg_count, _, _, pair = heapq.heappop(heap)
    
    return pair, heap


def _apply_merge_to_corpus(
        pair, 
        new_token_id, 
        pretoken_counts, 
        pair_to_pretokens, 
        counts, 
        heap, 
        tiebreak,
    ) -> None:
    """
    Apply a selected merge to every pretoken containing the pair.

    Mutates the weighted pretoken counts, global pair counts, reverse index,
    and candidate heap to reflect the merged corpus.
    """
    affected_pretoken_counts = {pretoken: pretoken_counts[pretoken]
                                for pretoken in pair_to_pretokens[pair]}
    for old_pretoken, frequency in affected_pretoken_counts.items():
        old_pair_counts = _count_pairs(old_pretoken)
        new_pretoken = tuple(_merge(old_pretoken, pair, new_token_id))
        new_pair_counts = _count_pairs(new_pretoken)
        affected_pairs = old_pair_counts.keys() | new_pair_counts.keys()
        for affected_pair in affected_pairs:
            new_occurrences = new_pair_counts.get(affected_pair, 0)
            old_occurrences = old_pair_counts.get(affected_pair, 0)
            delta_occurrences = new_occurrences - old_occurrences
            counts[affected_pair] += (delta_occurrences * frequency)
            if counts[affected_pair] == 0:
                del counts[affected_pair]
            else:
                _heappush(heap, affected_pair, counts[affected_pair], tiebreak)
            if old_occurrences > 0:
                pair_to_pretokens[affected_pair].remove(old_pretoken)
            if new_occurrences > 0:
                pair_to_pretokens[affected_pair].add(new_pretoken)
        del pretoken_counts[old_pretoken]
        pretoken_counts[new_pretoken] += frequency

def run_train_bpe(
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str]
        )-> BPETokenizerParams:
    # 1. Count corpus pretokens in parallel
    pretoken_counts = _count_pretokens_parallel(input_path, special_tokens)

    # 2. Initialize base vocabulary and merge budget
    merges: dict[tuple[int, int], int] = {}    # (idx1, idx2) -> new_idx
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}

    # Negate bytes so Python's min-heap prefers lexicographically larger tokens.
    # The trailing sentinel handles prefix cases: b"ab" should rank above b"a".
    # Since negated byte values are <= 0, sentinel 1 sorts after every byte.
    tiebreak: dict[int, list[int]] = {x: [-x, 1] for x in range(256)}
    for offset, special_token in enumerate(special_tokens):
        token_id = 256 + offset
        vocab[token_id] = special_token.encode('utf-8')
        tiebreak[token_id] = [-1 * byte_val for byte_val in vocab[token_id]] + [1]
    
    base_vocab_len = len(special_tokens) + 256
    if vocab_size < base_vocab_len:
        raise ValueError("desired vocab size too small.")
    num_merges = vocab_size - base_vocab_len

    # 3. Initialize pair counts, reverse index, and candidate priority heap
    counts, pair_to_pretokens = _count_adjacent(pretoken_counts)
    heap = []
    for pair, count in counts.items():
        _heappush(heap, pair, count, tiebreak)

    # 4. Learn each merge
    for i in range(num_merges):
        if not counts:
            break

        # Find the best pair
        pair, heap = _find_best_pair(heap, counts, tiebreak)

        # Record the selected merge and create its resulting vocabulary token.
        new_token_id = base_vocab_len + i
        merges[pair] = new_token_id
        vocab[new_token_id] = vocab[pair[0]] + vocab[pair[1]]
        tiebreak[new_token_id] = [-1 * b for b in list(vocab[new_token_id])] + [1]

        # Apply the selected merge only to affected pretokens, then update
        # pair counts, the reverse index, and candidate priorities.
        _apply_merge_to_corpus(pair, new_token_id, pretoken_counts, pair_to_pretokens, counts, heap, tiebreak)

    return BPETokenizerParams(vocab=vocab, merges=merges)

def save_params(params: BPETokenizerParams, output_dir: str | Path, file_name: str) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{file_name}.pkl"

    with output_path.open("wb") as f:
        pickle.dump(params, f)


def train_bpe_tinystories():
    filepath = "data/TinyStoriesV2-GPT4-train.txt"
    params = run_train_bpe(filepath, 10_000, ["<|endoftext|>"],)
    save_params(params, "tokenizer", "tiny_stories_params_opt")


def train_bpe_owt():
    filepath = "data/owt_train.txt"
    print("cwd:", os.getcwd())
    print("path:", Path(filepath).resolve())
    print("exists:", Path(filepath).exists())
    print("size:", Path(filepath).stat().st_size if Path(filepath).exists() else None)
    params = run_train_bpe(filepath, 32_000, ["<|endoftext|>"],)
    save_params(params, "tokenizer", "owt_params")


if __name__ == '__main__':
    iter = [1, 2, 3, 4]
    def practice_iterator(iter):
        yield from iter
    
    """
    a = practice_iterator(iter)
    print(a)
    for _ in range(5):
        print(next(a))
    """

    groups = [[1, 2], [3], [4, 5, 6]]

    def practice_iterator2(iter):
        for item in iter:
            if isinstance(item, list):
                yield from item
            else:
                yield item
        
    b = practice_iterator2(groups)
    for _ in range(7):
        print(next(b))




    """
    text_files = os.path.abspath(os.path.join(root_dir, 'cs336_basics/test_files/test2.txt'))
    owt_params = os.path.abspath(os.path.join(root_dir, 'tokenizer/owt_params.pkl'))
    owt_tokenizer = BPETokenizer.from_file(owt_params, ['<|endoftext|>'])
    with open(text_files, 'r') as file:
        text = file.read(1 * 1024 * 1024)
        encoded_file = owt_tokenizer.encode(text)
        print(text[:100])
    print("encoded file\n", encoded_file[:100])
    decoded_file = owt_tokenizer.decode(encoded_file)
    print("decoded_file\n", decoded_file[:100])
    assert text == decoded_file
    """
