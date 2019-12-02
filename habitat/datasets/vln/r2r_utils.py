#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import re
import gzip
import json
import string
from os import path
from collections import Counter
import numpy as np
from habitat.tasks.utils import heading_to_rotation


SCENE_ID = "mp3d/{scan}/{scan}.glb"
base_vocab = ['<pad>', '<unk>', '<s>', '</s>']
padding_idx = base_vocab.index('<pad>')


def make_id(path_id, instr_id):
    return str(path_id) + "_" + str(instr_id)


def read_json(filename):
    data = []
    if path.exists(filename):
        with open(filename) as f:
            data = json.load(f)
    else:
        print("Unable to read file. File %s not found" % filename)
    return data


def save_gzip(filename, content):
    with gzip.open(filename, 'wb') as f:
        f.write(json.dumps(content).encode('utf-8'))


def load_dataset(split, data_path):
    assert split in ['train', 'val_seen', 'val_unseen', 'test']
    return read_json(data_path.format(split=split))


def load_datasets(splits, data_path):
    data = []
    for split in splits:
        data += load_dataset(split, data_path)
    return data


def load_connectivity(data_path):
    return read_json(data_path)


class Tokenizer(object):
    ''' Class to tokenize and encode a sentence. '''
    # Split on any non-alphanumeric character
    SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')

    def __init__(self, vocab=None, encoding_length=20):
        self.encoding_length = encoding_length
        self.vocab = vocab
        self.word_to_index = {}
        if vocab:
            for i, word in enumerate(vocab):
                self.word_to_index[word] = i

    def split_sentence(self, sentence):
        ''' Break sentence into a list of words and punctuation '''
        toks = []
        words = self.SENTENCE_SPLIT_REGEX.split(sentence.strip())
        for word in [w.strip().lower() for w in words if len(w.strip()) > 0]:
            # Break up any words containing punctuation only,
            # e.g. '!?', unless it is multiple full stops e.g. '..'
            if all(c in string.punctuation for c in word) \
                    and not all(c in '.' for c in word):
                toks += list(word)
            else:
                toks.append(word)
        return toks

    def encode_sentence(self, sentence):
        if len(self.word_to_index) == 0:
            sys.exit('Tokenizer has no vocab')
        encoding = [self.word_to_index['<s>']]  # adding the start token
        for word in self.split_sentence(sentence)[::-1]:  # reverse input sent
            if word in self.word_to_index:
                encoding.append(self.word_to_index[word])
            else:
                encoding.append(self.word_to_index['<unk>'])
        encoding.append(self.word_to_index['</s>'])
        if len(encoding) < self.encoding_length:
            encoding += [self.word_to_index['<pad>']] * \
                    (self.encoding_length-len(encoding))
        return np.array(encoding[:self.encoding_length])

    def decode_sentence(self, encoding):
        sentence = []
        for ix in encoding:
            if ix == self.word_to_index['<pad>']:
                break
            else:
                sentence.append(self.vocab[ix])
        return " ".join(sentence[::-1])  # unreverse before output


def build_vocab(path, splits=['train'], min_count=5, start_vocab=base_vocab):
    '''
    Build a vocab, starting with base vocab containing a few useful tokens.
    '''
    count = Counter()
    t = Tokenizer()
    data = load_datasets(splits, path)
    for item in data:
        for instr in item['instructions']:
            count.update(t.split_sentence(instr))
    vocab = list(start_vocab)
    for word, num in count.most_common():
        if num >= min_count:
            vocab.append(word)
        else:
            break
    word2idx_dict = {v: i for i, v in enumerate(vocab)}
    return vocab, word2idx_dict


def serialize_r2r(config, splits=["train"], force=False) -> None:
    json_file_path = config.DATA_PATH[:-3]
    connectivity = load_connectivity(config.CONNECTIVITY_PATH)
    # Building both vocabularies Train and TrainVAL
    train_vocab, train_word2idx = build_vocab(json_file_path, splits=["train"])
    trainval_vocab, trainval_word2idx = \
        build_vocab(json_file_path, splits=["train", "val_seen", "val_unseen"])

    for split in splits:
        habitat_episodes = []
        if force or not path.exists(config.DATA_PATH.format(split=split)):
            data = load_dataset(split,
                                json_file_path.format(split=split))
            for episode in data:
                for i, instr in enumerate(episode["instructions"]):
                    viewpoint = episode["path"][0]
                    scan = episode["scan"]
                    distance = 0
                    if "distance" in episode:
                        distance = episode["distance"
                    habitat_episode = {
                        'episode_id': make_id(episode["path_id"], i),
                        'scene_id': SCENE_ID.format(scan=scan),
                        'start_position':
                            connectivity[scan][viewpoint]["start_position"],
                        'start_rotation':
                            heading_to_rotation(episode["heading"]),
                        'info':{"geodesic_distance": distance},
                        'path': episode["path"],
                        'instruction': instr,
                        'scan': scan
                    }
                    habitat_episodes.append(habitat_episode)

        if habitat_episodes:
            habitat_formatted_data = {
                "episodes": habitat_episodes,
                "train_vocab": {
                    "word_list": train_vocab,
                    "word2idx_dict": trainval_word2idx,
                    "itos": train_vocab,
                    "num_vocab": len(train_vocab),
                    'UNK_INDEX': 1,
                    'PAD_INDEX': 0
                },
                "trainval_vocab": {
                    "word_list": trainval_vocab,
                    "word2idx_dict": train_word2idx,
                    "itos": trainval_vocab,
                    "num_vocab": len(trainval_vocab),
                    'UNK_INDEX': 1,
                    'PAD_INDEX': 0
                },
            }

            print("writting", len(habitat_episodes))
            save_gzip(
                config.DATA_PATH.format(split=split),
                habitat_formatted_data
                )
    return 0
