import os
import sys
import json
import re
import argparse
import torchaudio
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer
from zmq import device
from utils.hangul_utils import join_jamos
from utils.audio_utils import convert_audio_pcm_to_wav_from_buffer
from transformers import AutoProcessor, AutoModelForCTC

def textprocessing(text):
    # pattern = re.compile('[^ ㄱ-ㅣ가-힣]+')
    # text = pattern.sub('', text)

    return text.replace("[PAD]", "")

def infer(model, processor, pcmaudio_path = None, wavaudio_path = None):
	model.cuda()
	model.eval()

	if pcmaudio_path:
		pcmdata, wavdata = convert_audio_pcm_to_wav_from_buffer(pcmaudio_path)

	elif wavaudio_path:
		wavdata, samplerate = torchaudio.load(wavaudio_path)
		wavdata = wavdata[0]

	features = processor(
		wavdata, 
		sampling_rate=16000, 
		padding=True,
		max_length=800000, 
		truncation=True,
		return_attention_mask=True,
		return_tensors="pt",
	)

	inputs = {
		'input_values': features['input_values'], 
		'attention_mask': features['attention_mask']
	}

	inputs = {k: v.cuda() for k, v in inputs.items()}

	outputs = model(**inputs, output_attentions=True)

	logits = outputs.logits
	predicted_ids = logits.argmax(axis=-1)
	text = processor.decode(predicted_ids[0])
	# stt_result = textprocessing(transcription) 

	return text

def main():
	device_id = "0"
	os.environ['CUDA_VISIBLE_DEVICES'] = device_id

	with open('test2.json', 'r') as testfile:
		testdata = json.load(testfile)[:15000]

	corpus_dir = "../dataset/Train_dataset"
	
	model_name_or_path = "models/char_300m_korean_1epoch_3e-4_30percent/checkpoint-18000"
	model = Wav2Vec2ForCTC.from_pretrained(model_name_or_path)
	processor = AutoProcessor.from_pretrained("w11wo/wav2vec2-xls-r-300m-korean")

	model_out = []

	for sample in tqdm(testdata):
		pcm_path = os.path.join(corpus_dir, sample['filepath'])
		inf = infer(model=model, processor=processor, pcmaudio_path=pcm_path)

		model_out.append(
			{
				#'id': sample['id'],
				'transcript': sample['transcript'],
				'inference': join_jamos(inf)
			}
		)

	with open(f'inference_{device_id}.json', 'w') as f:
		json.dump(model_out, f, indent='\t', ensure_ascii=False)

if __name__ == "__main__":
	main()
