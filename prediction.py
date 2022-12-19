# -*- coding: utf-8 -*- 
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer
import re

def clean_up(transcription):
    hangul = re.compile('[^ ㄱ-ㅣ가-힣]+')
    result = hangul.sub('', transcription)
    return result

# 모델명을 적어라
model_name = "model_name"

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
model = Wav2Vec2ForCTC.from_pretrained(model_name)
# wav 파일은 코드와 동일 위치에 놓을것
speech_array, sampling_rate = torchaudio.load("음성파일.wav")
feat = processor(speech_array[0], 
                            sampling_rate=16000, 
                            padding=True,
                            max_length=800000, 
                            truncation=True,
                            return_attention_mask=True,
                            return_tensors="pt",
                            pad_token_id=49
                            )
input = {'input_values': feat['input_values'],'attention_mask':feat['attention_mask']}

outputs = model(**input)
logits = outputs.logits
predicted_ids = logits.argmax(axis=-1)
transcription = processor.decode(predicted_ids[0])
stt_result = clean_up(transcription)

print(stt_result)