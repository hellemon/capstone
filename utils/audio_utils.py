import os
import sys
import json
import argparse
import numpy as np
from tqdm import tqdm

import ffmpeg
import librosa
import soundfile as sf


def convert_audio_pcm_to_wav(path):
	data, samplerate = sf.read(
		path, 
		channels=1, 
		samplerate=16000,
		format='RAW', 
		subtype='PCM_16'
	)

	return data, samplerate


def convert_audio_pcm_to_wav_from_buffer(path):
	with open(path, 'rb') as pcmfile:
		buf = pcmfile.read()
		pcmdata = np.frombuffer(buf, dtype = 'int16')
		wavdata = librosa.util.buf_to_float(pcmdata, 2)

	return pcmdata, wavdata


def convert_audio_wav_to_pcm(in_filename, **input_kwargs):
	try:
		out, err = (ffmpeg
			.input(in_filename, **input_kwargs)
			.output('-', format='s16le', acodec='pcm_s16le', ac=1, ar='16k')
			.overwrite_output()
			.run(capture_stdout=True, capture_stderr=True)
		)

	except ffmpeg.Error as e:
		print("Error")
		print(e.stderr, file=sys.stderr)
		sys.exit(1)

	return out
