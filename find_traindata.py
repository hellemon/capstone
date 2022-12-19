import os
import sys
import json
import random
import argparse
from tqdm import tqdm





def get_data_patial(corpus_dir, datapath, ratio=0.3):
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

        sample['filepath']

    return data

def split(corpus_dir, datapath):
    with open (datapath, "r", encoding='UTF-8') as f:
        data = json.load(f)
    with open ("valid+test_Data.json", "r") as f2:
        data2 = json.load(f2)

    for sample in tqdm(data, ascii=True):
        for sample2 in data2:
            if sample["filepath"] == sample2["filepath"]:
                del sample['filepath']
                del sample['transcript']
                break

    return data
        

test = split(corpus_dir="../dataset/Train_dataset", datapath="transcripts.json")

# train에서 사용된 데이터를 제외한다.
with open("train.json", 'w', encoding='utf-8') as file:
    json.dump(test, file, indent="\t")

