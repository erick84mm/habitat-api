#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import re
import gzip
import json
import math
import string
import numpy as np
import networkx as nx
import habitat_sim

from os import path
from collections import Counter
from habitat.tasks.utils import heading_to_rotation
from habitat_sim.utils.common import quat_from_two_vectors, quat_rotate_vector
from transformers.tokenization import BertTokenizer


SCENE_ID = "mp3d/{scan}/{scan}.glb"
base_vocab = ['<pad>', '<unk>', '<s>', '</s>']
padding_idx = base_vocab.index('<pad>')

def load_nav_graph(data):
    ''' Load connectivity graph for each scan '''

    def distance(pose1, pose2):
        ''' Euclidean distance between two graph poses '''
        return ((pose1['pose'][3]-pose2['pose'][3])**2\
          + (pose1['pose'][7]-pose2['pose'][7])**2\
          + (pose1['pose'][11]-pose2['pose'][11])**2)**0.5

    G = nx.Graph()
    positions = {}
    for i,item in enumerate(data):
        if item['included']:
            for j,conn in enumerate(item['unobstructed']):
                if conn and data[j]['included']:
                    positions[item['image_id']] = np.array([item['pose'][3],
                            item['pose'][7], item['pose'][11]])
                    assert data[j]['unobstructed'][i], 'Graph should be undirected'
                    G.add_edge(item['image_id'],data[j]['image_id'],weight=distance(item,data[j]))
    nx.set_node_attributes(G, values=positions, name='position')
    return G

def make_id(path_id, instr_id):
    return str(path_id) + "_" + str(instr_id)

def read(filename):
    data = []
    if path.exists(filename):
        with open(filename) as f:
            for line in f:
                data.append(line.strip())
    else:
        print("Unable to read file. File %s not found" % filename)
    return data


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


def load_connectivity(connectivity_path, scenes=None):
    file_format = connectivity_path + "{}_connectivity.json"
    if scenes:
        scans = scenes
    else:
        scans = read(connectivity_path + "scans.txt")
    connectivity = {}
    for scan in scans:
        data = read_json(file_format.format(scan))
        G = load_nav_graph(data)
        distances = dict(nx.all_pairs_dijkstra_path_length(G))
        paths = dict(nx.all_pairs_dijkstra_path(G))

        positions = {}
        visibility = {}
        idxtoid = {}
        idtoidx = {}
        for i, item in enumerate(data):
            idxtoid[i] = item['image_id']
            idtoidx[item['image_id']] = i
            pt_mp3d = np.array([item['pose'][3],
                item['pose'][7], item['pose'][11] - item['height']])

            q_habitat_mp3d = quat_from_two_vectors(
                np.array([0.0, 0.0, -1.0]),
                habitat_sim.geo.GRAVITY
            )
            pt_habitat = quat_rotate_vector(q_habitat_mp3d, pt_mp3d)
            positions[item['image_id']] = pt_habitat.tolist()
            visibility[item['image_id']] = {
                "included": item["included"],
                "visible": item["visible"],
                "unobstructed": item["unobstructed"],
            }

        connectivity[scan] = {
                "viewpoints": positions,
                "distances": distances,
                "paths": paths,
                "visibility": visibility,
                "idxtoid": idxtoid,
                "idtoidx": idtoidx,
        }

    return connectivity


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

def normalize_heading(heading):
    # Matterport goes from 0 to 2pi going clock wise.
    # Habitat goes from 0 - pi going counter clock wise.
    # Also habitat goes from 0 to - pi clock wise.

    if heading > np.pi:
        return 2 * np.pi - heading
    return -heading

class R2RSerializer:
    def __init__(config):
        json_file_path = config.DATA_PATH[:-3]
        self.connectivity = load_connectivity(config.CONNECTIVITY_PATH)
        train_vocab, train_word2idx = build_vocab(json_file_path, splits=["train"])
        trainval_vocab, trainval_word2idx = \
            build_vocab(json_file_path, splits=["train", "val_seen", "val_unseen"])
        if config.TOKENIZER == "BertTokenizer":
            self.tokenizer = BertTokenizer.from_pretrained(
                                 config.BERT_PRE_TRAINED_MODEL, do_lower_case=True
                             )
        else:
            self.tokenizer = Tokenizer(trainval_vocab, config.MAX_TOKEN_LENGTH)

    def _load_data(self):


        return

    def serialize_r2r(self, splits=["train"]) -> None:

        return


    def tokenize(text, max_length=16):
        """Tokenizes the instructions.
        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_index in embedding
        """
        for entry in self.entries:
            tokens = self._tokenizer.tokenize(entry["question"])
            tokens = ["[CLS]"] + tokens + ["[SEP]"]

            tokens = [
                self._tokenizer.vocab.get(w, self._tokenizer.vocab["[UNK]"])
                for w in tokens
            ]

            tokens = tokens[:max_length]
            segment_ids = [0] * len(tokens)
            input_mask = [1] * len(tokens)

            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self._padding_index] * (max_length - len(tokens))
                tokens = tokens + padding
                input_mask += padding
                segment_ids += padding

            assert_eq(len(tokens), max_length)
            entry["q_token"] = tokens
            entry["q_input_mask"] = input_mask
            entry["q_segment_ids"] = segment_ids

def tokenize_bert(text, padding=True, max_length=128, padding_index=0):
    print("tokenize")
    berttokenizer = BertTokenizer.from_pretrained(
                         "bert-base-uncased", do_lower_case=True
                     )
    print("initializing")
    tokens = berttokenizer.tokenize(text)
    tokens = ["[CLS]"] + tokens + ["[SEP]"]

    tokens = [
        berttokenizer.vocab.get(w, berttokenizer.vocab["[UNK]"])
        for w in tokens
    ]

    tokens = tokens[:max_length]
    segment_ids = [0] * len(tokens)
    input_mask = [1] * len(tokens)

    if len(tokens) < max_length:
        # Note here we pad in front of the sentence
        padding = [padding_index] * (max_length - len(tokens))
        tokens = tokens + padding
        input_mask += padding

    return tokens, input_mask

def serialize_r2r(config, splits=["train"], force=False) -> None:
    print("Starting serialization")
    json_file_path = config.DATA_PATH[:-3]
    connectivity = load_connectivity(config.CONNECTIVITY_PATH)
    print("connectivity loaded")
    # Building both vocabularies Train and TrainVAL
    train_vocab, train_word2idx = build_vocab(json_file_path, splits=["train"])
    trainval_vocab, trainval_word2idx = \
        build_vocab(json_file_path, splits=["train", "val_seen", "val_unseen"])
    tokenizer = Tokenizer(trainval_vocab, 100)
    print("vocabulary loaded")


    for split in splits:
        print("running split %s" % split)
        habitat_episodes = []
        scenes = []
        if force or not path.exists(config.DATA_PATH.format(split=split)):
            data = load_dataset(split,
                                json_file_path.format(split=split))
            print("Dataset loaded")
            for episode in data:
                for i, instr in enumerate(episode["instructions"]):
                    scan = episode["scan"]
                    scenes.append(scan)
                    goals = []
                    for vp in episode["path"]:
                        goals.append(connectivity[scan]["idtoidx"][vp])
                    viewpoint = episode["path"][0]
                    distance = 0
                    heading = normalize_heading(episode["heading"])
                    tokens, mask = tokenize_bert(instr)

                    if "distance" in episode:
                        distance = episode["distance"]
                    habitat_episode = {
                        'episode_id': make_id(episode["path_id"], i),
                        'scene_id': SCENE_ID.format(scan=scan),
                        'start_position':
                            connectivity[scan]["viewpoints"][viewpoint],
                        'start_rotation':
                            heading_to_rotation(heading),
                        'info': {"geodesic_distance": distance},
                        'goals': goals,
                        'instruction': instr,
                        'instruction_encoding': tokens,
                        'mask': mask,
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
                "scenes":list(set(scenes))
            }

            print("writting", len(habitat_episodes))
            save_gzip(
                config.DATA_PATH.format(split=split),
                habitat_formatted_data
                )

            with open(config.DATA_PATH.format(split=split)[:-8] + "_test.json", "w+") as outfile:
                json.dump(habitat_formatted_data, outfile)
    return 0
