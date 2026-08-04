import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
import pickle
from tokenizer import BPETokenizerParams

def _load(input_path):
    with open(input_path, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data

def get_largest_vocab(vocab, n):
    vocab = sorted([(k, v) for k, v in vocab.items()], key=lambda x: -len(x[1]))
    return [v.decode('utf-8', errors='backslashreplace') for k, v in vocab[:n]]


if __name__ == '__main__':
    owt_path = os.path.join(script_dir, '../tokenizer/owt_params.pkl')
    tiny_stories_path = os.path.join(script_dir, '../tokenizer/tiny_stories_params.pkl')

    data = _load(owt_path)
    n = 25
    owt_vocab = get_largest_vocab(data.vocab, n)
    print("OWT Top {n}")
    print("-" * 50)
    for i, v in enumerate(owt_vocab):
        print(f"{i}. {v}")

    data = _load(tiny_stories_path)
    ts_vocab = get_largest_vocab(data.vocab, n)
    print("TS Top {n}")
    print("-" * 50)
    for i, v in enumerate(ts_vocab):
        print(f"{i}. {v}")