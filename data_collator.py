import os
import sys
import json
import random
import argparse
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from utils.audio_utils import convert_audio_pcm_to_wav_from_buffer

import numpy as np
import torch
import torchaudio
import ast
from transformers import Wav2Vec2Processor
from torch.nn.utils.rnn import pad_sequence

def get_data_graphemes(corpus_dir, datapath):
    with open(datapath, 'r') as datafile:
        data = json.load(datafile)

    for sample in tqdm(data, desc=datapath):
        # sample 예시
        # {
        #     "id": 0,
        #     "filepath": "KsponSpeech_01/KsponSpeech_0001/KsponSpeech_000001.pcm",
        #     "transcript": "아 몬 소리야 그건 또",
        #     "graphemes": "ㅇ ㅏ | ㅁ ㅗ ㄴ | ㅅ ㅗ ㄹ ㅣ ㅇ ㅑ | ㄱ ㅡ ㄱ ㅓ ㄴ | ㄸ ㅗ",
        #     "labels": "101 115 4 91 123 75 4 97 123 82 135 101 117 4 70 133 70 119 75 4 80 123"
        # },

        audiopath = os.path.join(corpus_dir, sample['filepath'])
        wavdata, sr = torchaudio.load(audiopath)

        # pcmdata, wavdata = convert_audio_pcm_to_wav_from_buffer(audiopath)
        # wavdata = np.load("/home/leej/research/nc/xlsr/data/npy/" + str(sample['id']) + ".npy")

        sample['input_values'] = wavdata[0]
        # sample['labels'] = sample['graphemes']
        
        #del sample['id']
        del sample['filepath']
        del sample['transcript']
        #del sample['graphemes']

    return data

def get_data_graphemes_30percent(corpus_dir, datapath):
    with open(datapath, 'r') as datafile:
        data = json.load(datafile)
        random.seed(42)
        data = random.sample(data, int(len(data)*0.3))

    for sample in tqdm(data, desc=datapath):
        # sample 예시
        # {
        #     "id": 0,
        #     "filepath": "KsponSpeech_01/KsponSpeech_0001/KsponSpeech_000001.pcm",
        #     "transcript": "아 몬 소리야 그건 또",
        #     "graphemes": "ㅇ ㅏ | ㅁ ㅗ ㄴ | ㅅ ㅗ ㄹ ㅣ ㅇ ㅑ | ㄱ ㅡ ㄱ ㅓ ㄴ | ㄸ ㅗ",
        #     "labels": "101 115 4 91 123 75 4 97 123 82 135 101 117 4 70 133 70 119 75 4 80 123"
        # },

        audiopath = os.path.join(corpus_dir, sample['filepath'])
        wavdata, sr = torchaudio.load(audiopath)

        # pcmdata, wavdata = convert_audio_pcm_to_wav_from_buffer(audiopath)
        # wavdata = np.load("/home/leej/research/nc/xlsr/data/npy/" + str(sample['id']) + ".npy")

        sample['input_values'] = wavdata[0]
        # sample['labels'] = sample['graphemes']
        
        #del sample['id']
        del sample['filepath']
        del sample['transcript']
        #del sample['graphemes']

    return data

def get_data(corpus_dir, datapath):
    with open(datapath, 'r') as datafile:
        data = json.load(datafile)

    for sample in tqdm(data, desc=datapath):
        # sample 예시
        # {
        #     "id": 0,
        #     "filepath": "KsponSpeech_01/KsponSpeech_0001/KsponSpeech_000001.pcm",
        #     "transcript": "아 몬 소리야 그건 또",
        #     "graphemes": "ㅇ ㅏ | ㅁ ㅗ ㄴ | ㅅ ㅗ ㄹ ㅣ ㅇ ㅑ | ㄱ ㅡ ㄱ ㅓ ㄴ | ㄸ ㅗ",
        #     "labels": "101 115 4 91 123 75 4 97 123 82 135 101 117 4 70 133 70 119 75 4 80 123"
        # },

        audiopath = os.path.join(corpus_dir, sample['filepath'])
        wavdata, sr = torchaudio.load(audiopath)

        # pcmdata, wavdata = convert_audio_pcm_to_wav_from_buffer(audiopath)
        # wavdata = np.load("/home/leej/research/nc/xlsr/data/npy/" + str(sample['id']) + ".npy")

        sample['input_values'] = wavdata[0]
        sample['labels'] = sample['transcript']
        
        #del sample['id']
        del sample['filepath']
        del sample['transcript']
        #del sample['graphemes']

    return data

def get_data_patial(corpus_dir, datapath, ratio=1.0):
    with open(datapath, 'r', encoding='UTF-8') as datafile:
        data = json.load(datafile)
        random.seed(42)
        data = random.sample(data, int(len(data)*ratio))

    for sample in tqdm(data, desc=datapath, ascii=True):
        # sample 예시
        # {
        #     "id": 0,
        #     "filepath": "KsponSpeech_01/KsponSpeech_0001/KsponSpeech_000001.pcm",
        #     "transcript": "아 몬 소리야 그건 또",
        #     "graphemes": "ㅇ ㅏ | ㅁ ㅗ ㄴ | ㅅ ㅗ ㄹ ㅣ ㅇ ㅑ | ㄱ ㅡ ㄱ ㅓ ㄴ | ㄸ ㅗ",
        #     "labels": "101 115 4 91 123 75 4 97 123 82 135 101 117 4 70 133 70 119 75 4 80 123"
        # },

        audiopath = os.path.join(corpus_dir, sample['filepath'])
        wavdata, sr = torchaudio.load(audiopath)

        # pcmdata, wavdata = convert_audio_pcm_to_wav_from_buffer(audiopath)
        # wavdata = np.load("/home/leej/research/nc/xlsr/data/npy/" + str(sample['id']) + ".npy")

        sample['input_values'] = wavdata[0]
        sample['labels'] = sample['transcript']
        
        #del sample['id']
        del sample['filepath']
        del sample['transcript']
        #del sample['graphemes']

    return data

@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    grapheme_label: bool
    char_label: bool
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def mapping(self, data):
        with self.processor.as_target_processor():
            # ret = self.processor("".join([i if i != '\x1b' else '|' for i in ast.literal_eval(data)])).input_ids
            ret = self.processor("".join([i if i != ' ' else '|' for i in data])).input_ids
            ret_torch = torch.tensor([int(0 if value is None else value) for value in ret])

        return ret_torch

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature['input_values']} for feature in features]
        
        # e.g. feature['labels'] = "ㅇㅏㄴㄴㅕㅇㅎㅏㅅㅔㅇㅛ"
        if not self.char_label:
            label_features = [{"input_ids": torch.tensor([int(label) for label in feature["labels"].split()])} for feature in features]
        elif self.char_label:
            label_features = [{"input_ids": self.mapping(feature["labels"])} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels

        return batch


# 자소 단위로 나누어지지 않은 경우 사용 -> 한번 하면 저장해놓기
class DataProc:
    def __init__(self, model_name):
        self.processor = Wav2Vec2Processor.from_pretrained(model_name, pad_token_id=49)

    def to_jaso(self, sentence):
        NO_JONGSUNG = ''
        CHOSUNGS = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
        JOONGSUNGS = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
        JONGSUNGS = [NO_JONGSUNG,  'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

        N_CHOSUNGS, N_JOONGSUNGS, N_JONGSUNGS = 19, 21, 28
        FIRST_HANGUL, LAST_HANGUL = 0xAC00, 0xD7A3 #'가', '힣'    
     
        result = []
        for char in sentence:
            if ord(char) < FIRST_HANGUL or ord(char) > LAST_HANGUL: 
                result.append('|')
            else:          
                code = ord(char) - FIRST_HANGUL
                jongsung_index = code % N_JONGSUNGS
                code //= N_JONGSUNGS
                joongsung_index = code % N_JOONGSUNGS
                code //= N_JOONGSUNGS
                chosung_index = code
                result.append(CHOSUNGS[chosung_index])
                result.append(JOONGSUNGS[joongsung_index])
                if jongsung_index!=0:
                    result.append(JONGSUNGS[jongsung_index])
                
        with self.processor.as_target_processor():
            ret = self.processor("".join(result))
        
        return ret.input_ids

    def prepare_dataset(self, df):
        """
        df.cols = ['audio', 'sentence', 'path']
        """
        df['label'] = df['sentence'].apply(self.to_jaso)
        return df