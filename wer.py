from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer
from datasets import load_dataset
import soundfile as sf
import torch
from jiwer import wer

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
	"w11wo/wav2vec2-xls-r-300m-korean",
    do_normalize = True,
    feature_extractor_type = "Wav2Vec2FeatureExtractor",
    feature_size = 1,
    padding_side = "right",
    padding_value = 0.0,
    return_attention_mask = True,
    sampling_rate = 16000
)

tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("w11wo/wav2vec2-xls-r-300m-korean")

processor = Wav2Vec2Processor(
	feature_extractor=feature_extractor, 
	tokenizer=tokenizer
)

model = Wav2Vec2ForCTC.from_pretrained("models/char_300m_korean_1epoch_3e-4_30percent/checkpoint-19000").to('cuda')

#ds = load_dataset("kresnik/zeroth_korean", "clean")
data_files = {"train": "train.json", "test": "test2.json"}

#여기서 별도로 지정을 안해줘서 train으로 생긴다 load_dataset의 기본값은 train
'''
data_files = {"train": "SQuAD_it-train.json", "test": "SQuAD_it-test.json"}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")
squad_it_dataset
'''
ds = load_dataset("json", data_files="test2.json")
test_ds = ds

def map_to_array(batch):
    speech, _ = sf.read("../dataset/Train_dataset/"+batch["filepath"])
    batch["speech"] = speech
    return batch

test_dst = test_ds.map(map_to_array)

def map_to_pred(batch):
    inputs = processor(batch["speech"], sampling_rate=16000, return_tensors="pt", padding="longest")
    input_values = inputs.input_values.to("cuda")
    #attention_mask = inputs.attention_mask.to("cuda")
    
    with torch.no_grad():
        #logits = model(input_values, attention_mask=attention_mask).logits
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    batch["transcription"] = transcription
    return batch

result = test_dst.map(map_to_pred, batched=True, batch_size=16, remove_columns=["speech"])
print(result)
print("WER:", wer(result["train"]['transcript'],result["train"]['transcription']))
