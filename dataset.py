"""
Preprocesses MIDI files
"""
import math
import numpy as np
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import Dataset

import numpy
import math
import random
from tqdm import tqdm
import multiprocessing
import itertools

from constants import *
from midi_io import load_midi
from util import *
import constants as const

class MusicDataset(Dataset):
    def __init__(self, data_files, transform=lambda x: x):
        """    
        Loads all MIDI files from provided files.
        """
        self.transform = transform
        self.seqs = []
        ignore_count = 0
        
        for f in tqdm(data_files):
            try:
                # Pad the sequence by an empty event
                seq = load_midi(f)
                if len(seq) < SEQ_LEN:
                    raise Exception('Ignoring {} because it is too short {}.'.format(f, len(seq)))
                # elif len(seq) > MAX_SEQ_LEN:
                #     raise Exception('Ignoring {} because it is too long {}.'.format(f, len(seq)))
                else:
                    seq = np.concatenate([[EOS], seq, [EOS]])
                    self.seqs.append(torch.from_numpy(seq).long())                    
            except Exception as e:
                print('Unable to load {}'.format(f), e)
                ignore_count += 1

        print('{} files ignored.'.format(ignore_count))
        print('Loaded {} MIDI file(s) with average length {}'.format(len(self.seqs), sum(len(s) for s in self.seqs) / len(self.seqs)))

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.transform(self.seqs[idx])


def transform(seq):
    # Apply random augmentations
    seq = transpose(seq)
    seq = stretch_sequence(seq, random.uniform(1.0, 1.25))
    return tensorize(seq)

def tensorize(seq):
    """ Converts a generator into a Torch LongTensor """
    return torch.LongTensor(list(seq))

def stretch_sequence(sequence, stretch_scale):
    """ Iterate through sequence and stretch each time shift event by a factor """
    # Accumulated time in seconds
    time_sum = 0
    seq_len = 0
    for i, evt in enumerate(sequence):
        if evt >= TIME_OFFSET and evt < VEL_OFFSET:
            # This is a time shift event
            # Convert time event to number of seconds
            # Then, accumulate the time
            time_sum += convert_time_evt_to_sec(evt)
        else:
            if i > 0:
                # Once there is a non time shift event, we take the
                # buffered time and add it with time stretch applied.
                for x in seconds_to_events(time_sum * stretch_scale):
                    yield x
                # Reset tracking variables
                time_sum = 0
            seq_len += 1
            yield evt

    # Edge case where last events are time shift events
    if time_sum > 0:
        for x in seconds_to_events(time_sum * stretch_scale):
            seq_len += 1
            yield x

    # Pad sequence with empty events if seq len not enough
    if seq_len < SEQ_LEN:
        for x in range(SEQ_LEN - seq_len):
            yield TIME_OFFSET
            
def transpose(sequence, amount=4):
    """ A generator that represents the sequence. """
    # Transpose by 4 semitones at most
    transpose = random.randint(-amount, amount)

    if transpose == 0:
        return sequence

    # Perform transposition (consider only notes)
    return (evt + transpose if (evt >= NOTE_ON_OFFSET and evt < TIME_OFFSET) else evt for evt in sequence)

def collate_fn(data):
    """
    Creates mini-batch tensors from the list of data.
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of sequences (LongTensor of shape (seq_len))
    Returns:
        seqs: tensor of shape (batch_size, seq_len).
    """
    # Sort a data list by sequence length (descending order).
    seqs = data
    seqs.sort(key=lambda x: len(x), reverse=True)
    lengths = [len(x) for x in seqs]

    seqs = pack_padded_sequence(
        pad_sequence(seqs, batch_first=True, padding_value=const.EOS),
        lengths,
        batch_first=True
    )
    return seqs

def get_tv_loaders():
    data_files = get_all_files(const.STYLES)
    train_files, val_files = validation_split(data_files)
    print('Training Files:', len(train_files), 'Validation Files:', len(val_files))
    return get_loader(train_files), get_loader(val_files)

def get_loader(files):
    ds = MusicDataset(files, transform)
    return torch.utils.data.DataLoader(
        dataset=ds, 
        batch_size=const.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )

def validation_split(seqs, split=0.2):
    """
    Splits the data iteration list into training and validation indices
    """
    # Shuffle sequences randomly
    r = list(range(len(seqs)))
    random.shuffle(r)

    num_val = int(math.ceil(len(r) * split))
    train_indicies = r[:-num_val]
    val_indicies = r[-num_val:]

    assert len(val_indicies) == num_val
    assert len(train_indicies) == len(r) - num_val

    train_seqs = [seqs[i] for i in train_indicies]
    val_seqs = [seqs[i] for i in val_indicies]

    return train_seqs, val_seqs
