import unicodedata
import re
import torch
import random


# start and end tokens
SOS_token = 0
EOS_token = 1

MAX_LENGTH = 10
ENG_PREFIXES = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def check_pair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(ENG_PREFIXES)


def filter_pairs(pairs):
    # file is english french, so must reverse when filtering
    return [p for p in pairs if check_pair(p)]


def prepare_data(lang1, lang2, simple_sentences=True):
    print('Loading data')
    with open(f'pytorch_tutorial/data/fra-eng/fra.txt') as f:
        pairs = [[normalizeString(s) for s in line.split('\t')[:2]][::-1] for line in f]
    print(f"example loaded pair  {pairs[0]}")
    print(f"total sentance pairs in data : {len(pairs)}")
    print('filtering data')
    if simple_sentences:
        filtered_pairs = filter_pairs(pairs)
        print(f"Filtered sentence pairs in data : {len(filtered_pairs)}")

    assert len(filter_pairs(filtered_pairs)) == len(filtered_pairs), "nothing should escape filtering!"
    l1 = Lang(lang1)
    l2 = Lang(lang2)

    for p in filtered_pairs:
        l1.add_sentence(p[0])
        l2.add_sentence(p[1])
    print(f'{l1.name} : {l1.n_words}')
    print(f'{l2.name} : {l2.n_words}')
    return l1, l2, filtered_pairs


class Lang:
    """Helper class to manage index <=> word translation"""

    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.index2word = {SOS_token: 'SOS', EOS_token: 'EOS'}
        self.word2count = {}  # for rare words
        self.n_words = 2

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.word2count[word] = 1
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'

    )


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    # adds a space before end of sentance punctuation
    s = re.sub(r"([.!?])", r" \1", s)
    # replaces all other punctuation or unusual characters with spaces
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def sentence2indexlist(lang, s):
    return [lang.word2index[w] for w in s.split(' ')]


def sentence2tensor(lang, s, device):
    indexes = sentence2indexlist(lang, s)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def pair2tensors(l1, l2, pair, device):
    input_tensor = sentence2tensor(l1, pair[0], device)
    target_tensor = sentence2tensor(l2, pair[1], device)
    return (input_tensor, target_tensor)




def timedelta_string(delta_time):
    days = delta_time.days
    hours = delta_time.seconds // 3600
    minutes = delta_time.seconds % 3600 // 60
    seconds = delta_time.seconds % 60 + delta_time.microseconds / 1e6
    return f"{days:3>}:{hours:02}:{minutes:02}:{seconds:05.2f}"

