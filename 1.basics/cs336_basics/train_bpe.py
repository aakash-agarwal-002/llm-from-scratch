from cs336_basics.pretokenization_example import find_chunk_boundaries
from collections import Counter
import regex as re
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
import time

# ---- Precompile regex ONCE per process ----
def init_worker():
    global RX
    RX = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
        re.UNICODE
    )


# ---- Process one chunk (BIG TASK) ----
def process_chunk(args):
    input_path, start, end = args

    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    # split on special token
    docs = chunk.split("<|endoftext|>")

    local_counter = Counter()

    for doc in docs:
        tokens = RX.findall(doc)
        tokens = [
            tuple(bytes([b]) for b in token.encode("utf-8"))
            for token in tokens
        ]
        
        local_counter.update(tokens)

    return local_counter


def pairs_in_word(word : tuple[bytes,...]):
    counts: dict[tuple[bytes, bytes], int] = {}
    for i,j in zip(word[:-1],word[1:]):
        counts[(i,j)] = counts.get((i,j), 0) + 1
    return counts
        

def build_pair_stats(word_freq : dict[tuple[bytes,...], int]) -> tuple[dict[tuple[bytes, bytes], int], dict[tuple[bytes, bytes], set[tuple[bytes,...]]]]:
    pair_counts = defaultdict(int)
    pair_to_words = defaultdict(set)
    for word, freq in word_freq.items():
        if len(word)<2:
            continue
        local_pairs = pairs_in_word(word)
        for pair, count in local_pairs.items():
            pair_counts[pair] += count*freq
            pair_to_words[pair].add(word)
    return pair_counts, pair_to_words

def apply_merge(merge_pair : tuple[bytes,bytes], word : tuple[bytes,...],new_token : bytes)-> tuple[bytes,...]:
    new_word = []
    i=0
    while i<len(word):
        if i<len(word)-1 and (word[i],word[i+1]) == merge_pair:
            new_word.append(new_token)
            i+=1
        else:
            new_word.append(word[i])
        i+=1
    return tuple(new_word)
            


def remove_word_contrib(word : tuple[bytes,...], freq : int,pair_counts: dict[tuple[bytes,bytes],int], pair_to_words : dict[tuple[bytes,bytes],set[tuple[bytes,...]]]):
    """
    we remove contribution of each word in affected word.
    1. remove its affect from pair_counts for all pair in word
    2. remove its affect from pair_to_words because these pairs have in there set this affected word 
    """
    local_pairs = pairs_in_word(word)
    for pair, occ in local_pairs.items():
        
        if pair in pair_counts:
            pair_counts[pair]-= occ*freq
            if pair_counts[pair] <= 0:
                del pair_counts[pair]
    
        if pair in pair_to_words:
            pair_to_words[pair].discard(word)
            if not pair_to_words[pair]:
                del pair_to_words[pair]

def add_word_contrib(word : tuple[bytes,...], freq : int,pair_counts: dict[tuple[bytes,bytes],int], pair_to_words : dict[tuple[bytes,bytes],set[tuple[bytes,...]]]):
    """
    we add back contribution of each word in affected word.
    1. add back its affect on pair_counts for all pair in word
    2. add back its affect on pair_to_words because these pairs have in there set this affected word 
    """
    local_pairs = pairs_in_word(word)
    for pair, occ in local_pairs.items():
        pair_counts[pair]+=occ*freq
        pair_to_words[pair].add(word)
        


def train_bpe(input_path:str,vocab_size: int, special_tokens: list[str], num_processes = 4):
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    next_id = 256
    for token in special_tokens:
        vocab[next_id] = token.encode("utf-8")
        next_id += 1

    # find chunk boundaries
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"")

    tasks = [
        (input_path, start, end)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    start_time = time.time()

    with ProcessPoolExecutor(max_workers=num_processes, initializer=init_worker) as executor:
        results = list(executor.map(process_chunk, tasks))

    word_freq = Counter()
    for r in results:
        word_freq.update(r)

    end_time = time.time()

    print(f"Parallel time: {end_time - start_time:.2f}s")
    print(f"Unique pre-tokens: {len(word_freq)}")

    pair_counts, pair_to_words = build_pair_stats(word_freq)
    merges : list[tuple[bytes, bytes]] = []
    
    while next_id < vocab_size:
        (a, b), best_count = max(pair_counts.items(), key=lambda x: (x[1],x[0]))
        new_token = a + b
        merges.append((a, b))
        vocab[next_id] = new_token
        next_id+=1
        
        affected_words = list(pair_to_words[(a,b)])
        if not affected_words: 
            continue
        
        add_back = defaultdict(int)
        for word in affected_words:
            remove_word_contrib(word=word,freq=word_freq[word],pair_counts=pair_counts,pair_to_words=pair_to_words)
            add_back[apply_merge((a, b),word,new_token)]+=word_freq[word]
            del word_freq[word]
        
        for word,freq in add_back.items():
            word_freq[word] = word_freq.get(word,0)+freq
            add_word_contrib(word=word,freq=freq,pair_counts=pair_counts,pair_to_words=pair_to_words)

    # print(vocab)
    # print(merges)
    return vocab, merges
if __name__ == "__main__":
    train_bpe(
        input_path="../data/TinyStoriesV2-GPT4-valid.txt",
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
        num_processes = 8
    )
