# 복지분야 콜센터 데이터를 이용한 XLS-R Finetunning
복지분야 콜센터 데이터를 사용하여 사전학습된 XLS-R 모델에 Fine-Tunning을 진행 하였습니다
사용된 사전학습 모델은 Hugging Face에 있는 다음과 같은 모델입니다.
https://huggingface.co/w11wo/wav2vec2-xls-r-300m-korean

## Summary
+ AI-Hub에 있는 복지분야 콜센터 데이터 (180만개)를 활용하여 사전학습 모델을 Fine-Tunning 하고 이를 통해 AI 챗봇서비스를 발굴하는 프로젝트 내용입니다.
+ XLS-R-300M 모델을 선택한 이유는 트랜스포머를 기반으로한 거대 모델이기 때문입니다. 또한 사전학습이 된 모델이 존재하며, 1B, 2B모델의 경우에는 사양이 더 높을 필요가 있을 뿐더러 사전학습 모델이 존재하지 않기 때문입니다.
+ 해당 사전학습 모델과 Process, tokenizer를 사용한 이유는 초성,중성,종성에 해당하는 자모를 사용하는 것이 정확도를 더 높히고 효율적이겠지만, 아직 부족하여서 그렇습니다.

## 초기 환경 (실행환경)
-운영체제- 
1. Ubuntu 18.04.4 LTS

-하드웨어 사양-
1. VRAM 10GB 이상의 GPU 환경 (Batch size가 4를 넘어가면 cuda error)
2. RAM : 250BG(사용하고자하는 데이터셋의 크기 보다 큰 RAM이 필요) 
3. CUDA version (10.2)

-소프트웨어-
1. Anaconda
2. visual studio code

## 설치해야 하는 library
1. numpy
2. pytorch (gpu version) 1.12.1
3. tqdm
4. datasets
5. transformers
6. torchaudio
7. zmq
8. jiwer
9. ffmpeg


## 한번에 설치 Pytorch 제외
        pip install -r requirements.txt


## Preprocessing
### 전체 데이터셋 txt 파일만들기 
+ output_unit은 kospeech가 제공하는 3가지 방식, preprocess_mode는 퍼센트 표기 방식
+ preprocess.py 파일을 실행하기 위해서 "오디오파일경로"+"\t"로 되어 있습니다.
+ 전처리 후 학습/평가 등에 필요한 transcript.txt가 생기게 됩니다.

        python preprocessing/main.py --dataset_path $dataset_path --vocab_dest $vacab_dict_destination --output_unit 'character' --preprocess_mode 'phonetic' 

### txt 파일을 json 형태로 변경
+ 저의 경우에는 전처리 단계에서 시도를 하다가, 결국 손으로 수정하였습니다.
+ change_string.py를 사용하여서 중괄호를 json 형태에 맞게끔 입력해주시면 됩니다.
+ 파일의 위치, 결과의 경우 규칙성이 있기 때문에 해당 부분의 괄호를 넣는건 오래걸리지는 않습니다.
+ json 형태로 완성시에는 vscode에서 파일 확장자를 txt -> json형태로 변경해주면 동작하게 됩니다.

## Train
+ Train.py에서 다음 부분에 대해 값을 지정해주시면 됩니다.
        
        model = get_model(model_name_or_path="w11wo/wav2vec2-xls-r-300m-korean")
        
        valid = get_data_patial(corpus_dir="../dataset/Test_dataset", datapath="val_transcripts.json", ratio=1)
        
        train = get_data_patial(corpus_dir="../dataset/Train_dataset", datapath="transcripts.json", ratio=0.3)
+ get_model의 경우 hugging Face에서 가져오게 될 사전학습 모델을 넣는 곳입니다.
+ get_data_partial 함수는 전체 데이터에서 ratio 비율만큼 값을 가져오게 됩니다. (1로 선택시 전체 데이터), corpus_dir은 json에서 파일의 위치의 앞 부분을 입력해 주시면 됩니다. datapath는 학습/평가 등 하고자 하는 json파일
+ 변경 후 실행 

        python train.py


## inference
### 여러개의 데이터를 전사
+ 여러개의 데이터를 inference 할 경우 다음과 같습니다. inference.py 코드에서 해당 부분(61 line)에 json 파일을 전사하고자 하는 json파일로 변경 후 실행 하면 전사된 모델이 나옵니다.

        	with open('test2.json', 'r') as testfile:
		        testdata = json.load(testfile)[:15000]

+ 모델에 대한 변경은 inference.py 에서 해당 부분(66 line)에서 바꾸어 주시면 됩니다.
        	
            model_name_or_path = "models/char_300m_korean_1epoch_3e-4_30percent/checkpoint-18000"
	        model = Wav2Vec2ForCTC.from_pretrained(model_name_or_path)

+ 변경 이후 코드 실행하면 json 형태로 결과 값이 나오게 됩니다.


#### 결과물 예시
inference_0.json 이란 이름의 json 파일이 생성이 되며 내용은 다음과 같습니다.
                
                {
		        "transcript": "골수기증은 아무나 할 수 있는 것인지요",
		        "inference": "골 수이증은 아무나 할 수 있는 것인지요
	        },
위와 같은 형태가 반복된 정답과 출력값으로 이어진 json파일이 생성된다

### 단일 데이터 전사
+ 하나의 데이터를 전사하고자 할 경우 prediction.py를 사용하시면 됩니다.
    해당 부분에서 오디오 파일을 바꾸게 될경우 해당 음성파일에 대한 전사가 print 되어 나오게 됩니다.

        speech_array, sampling_rate = torchaudio.load("HOS11000511442B024.wav")

#### 단일 데이터 전사 결과물
+ 입력한 wav 파일에 대한 내용을 print합니다.

### wer 측정
+ 측정하고자 하는 모델은 wer.py (25 line)다음 위치에서 변경을 할 수 있습니다. 아래 예시는 Hugging Face의 사전학습 모델
        
        model = Wav2Vec2ForCTC.from_pretrained("w11wo/wav2vec2-xls-r-300m-korean").to('cuda')

+ wer측정의 경우 진행하고자 하는 json 파일을 필요로 합니다. 해당 json 파일을 다음과 같은 형태로 가져 온 뒤(36 line) 
        
        ds = load_dataset("json", data_files="trash.json")

+ 변경을 다 마치고, 실행하게 되면 wer의 결과값이 나오게 됩니다.

## 추가적인 부연 설명
+ 제가 사용한 json 파일에서는 저장 경로가 KsponSpeech로 되어 있는데 해당 이유는 Kospeech전처리를 하는 과정에서 해당 코드가 KsponSpeech 즉 AIhub에서 제공하는 한국어 대화 폴더에 맞춰서 만들어져 있기 때문입니다. 저는 해당 틀에 맞춰서 Transcript를 만들었지만 데이터셋을 모으는 폴더를 KsponSpeech로 하거나 혹은 코드를 바꿔서 폴더명을 바꿔서 사용할 수도 있을 것입니다. 
저는 데이터셋을 저장하는 폴더명을 KsponSpeech로 해서 만든 후 change_string.py를 이용하여서 추후 폴더명을 바꾸거나 하는 것을 권장합니다.