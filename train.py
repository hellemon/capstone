from datasets import load_metric, load_from_disk
from data_collator import DataCollatorCTCWithPadding, get_data_graphemes, get_data, get_data_patial
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer
from transformers import TrainingArguments, Trainer
from tqdm import tqdm
import torch
import ast
import numpy as np
import os

# os.environ['CUDA_LAUNCH_BLOCKING']='1'
os.environ['CUDA_VISIBLE_DEVICES']="2,3"

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
# tokenizer = Wav2Vec2CTCTokenizer(
#     # vocab_file = "./data/vocab/vocab_phoneme.json",
#     vocab_file = "vocabs_char.json",
# 	bos_token = '<s>',
# 	eos_token = '</s>',
#     unk_token="[UNK]",
#     pad_token="[PAD]",
#     word_delimiter_token="|"
# )

processor = Wav2Vec2Processor(
	feature_extractor=feature_extractor, 
	tokenizer=tokenizer
)

wer_metric = load_metric("wer")
# data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True, grapheme_label=True, char_label=False)
data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True, grapheme_label=False, char_label=True)

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

def get_model(model_name_or_path): 
    model = Wav2Vec2ForCTC.from_pretrained(
        model_name_or_path, 
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.1,
        ctc_loss_reduction="mean", 
        pad_token_id=tokenizer.pad_token_id,
        vocab_size=tokenizer.vocab_size,
        ignore_mismatched_sizes=True
    )

    model.freeze_feature_extractor()
    model.gradient_checkpointing_enable()

    return model

def train_model(train, test, model):
    train_epochs = 1
    learning_rate = 3e-4
    lr = "3e-4"
    output_dir = f"./models/char_300m_korean_{train_epochs}epoch_{lr}_30percent"

    training_args = TrainingArguments(
        output_dir=output_dir,
        group_by_length=True,
        per_device_train_batch_size=4,
        # per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        do_eval=False,
        evaluation_strategy="no",
        num_train_epochs=train_epochs,
        gradient_checkpointing=True,
        fp16=True, # 원래 True -> 에러 후 False로 변경
        no_cuda=False,
        save_steps=1000,
        # eval_steps=200,
        logging_steps=50,
        learning_rate=learning_rate,
        # warmup_steps=300,
        warmup_ratio=0.06,
        # save_total_limit=5,
        # load_best_model_at_end=True,
        # metric_for_best_model='eval_loss',
        dataloader_num_workers=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train,
        eval_dataset=test,
        tokenizer=processor.feature_extractor,
        data_collator=data_collator,
    )

    print(f"build trainer on device {training_args.device} with {training_args.n_gpu} gpus")
    trainer.train()
    # trainer.train(os.path.join(output_dir, "checkpoint-4160"))
    trainer.save_model()
    processor.save_pretrained(output_dir)
    # torch.save(model.state_dict(), os.path.join(output_dir, 'model.pt'))

if __name__ == "__main__":
    
    torch.cuda.empty_cache()
    model = get_model(model_name_or_path="w11wo/wav2vec2-xls-r-300m-korean")
    valid = get_data_patial(corpus_dir="../dataset/Test_dataset", datapath="val_transcripts.json", ratio=1)
    train = get_data_patial(corpus_dir="../dataset/Train_dataset", datapath="transcripts.json", ratio=0.3)
    

    
    train_model(train, valid, model)
    