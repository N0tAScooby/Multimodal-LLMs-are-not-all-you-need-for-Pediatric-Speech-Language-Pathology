import os
import re
import torch
import pandas as pd
import numpy as np
import wandb
import jiwer
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Union, Optional
from datasets import load_dataset, Audio, Dataset, DatasetDict
from sklearn.model_selection import GroupShuffleSplit
from transformers import (
    WhisperProcessor, 
    WhisperForConditionalGeneration, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    BitsAndBytesConfig, 
    EarlyStoppingCallback,
    set_seed
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from audiomentations import Compose, AddGaussianNoise, PitchShift, OneOf


DATASET_ID = "SAA-Lab/SLPHelmUltraSuitePlus"
WANDB_PROJECT = "project_name"
OUTPUT_DIR_BASE = "./sweep-checkpoints"

from transformers import TrainerCallback

class WandBPredictionProgressCallback(TrainerCallback):
    def __init__(self, trainer, tokenizer, val_dataset, num_samples=5):
        super().__init__()
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.sample_dataset = val_dataset.select(range(min(num_samples, len(val_dataset))))

    def on_evaluate(self, args, state, control, **kwargs):
        
        
        current_epoch = int(round(state.epoch)) if state.epoch is not None else 0
        
        if current_epoch > 0 and current_epoch % 5 == 0:
            print(f"\n--- Logging Sample Predictions for Epoch {current_epoch} ---")
            
            
            predict_results = self.trainer.predict(self.sample_dataset)
            
            pred_ids = predict_results.predictions
            label_ids = predict_results.label_ids
            
            
            label_ids[label_ids == -100] = self.tokenizer.pad_token_id
            
            preds_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            labels_str = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

            
            my_table = wandb.Table(columns=["Epoch", "Ground Truth", "Prediction"])
            for label, pred in zip(labels_str, preds_str):
                my_table.add_data(current_epoch, label, pred)
            
            wandb.log({f"eval_predictions_epoch_{current_epoch}": my_table}, commit=False)




def load_and_split_data():
    print(f"Loading dataset {DATASET_ID}...")
    raw_dataset = load_dataset(DATASET_ID, split="train") 
    
    
    def is_valid_transcription(example):
        text = example["transcription"]
        
        
        if not text or len(text.strip()) == 0:
            return False
            
        text_lower = text.lower()
        
        clinical_tags = [
            "pre-test", 
            "post-test",
            "no-context", 
            "vowel", 
            "cardinal", 
            "linguolabial", 
            "alveolar",
            "fricative",
            "plosive",
            "consonant",
            "close-",
        ]
        
        
        if any(tag in text_lower for tag in clinical_tags):
            return False
            
        return True

    
    initial_count = len(raw_dataset)
    raw_dataset = raw_dataset.filter(is_valid_transcription)
    filtered_count = len(raw_dataset)
    print(f"Filtered out {initial_count - filtered_count} metadata samples. Remaining: {filtered_count}")
    
    
    
    print("Extracting Speaker IDs...")
    ds_meta = raw_dataset.cast_column("audio", Audio(decode=False))
    paths = [item['audio']['path'] for item in ds_meta]
    pattern = re.compile(r"^(?:test_|train_|val_|audio_)?([a-zA-Z0-9]+)[_\-\.]", re.IGNORECASE)
    speaker_ids = []
    for p in paths:
        filename = os.path.basename(p)
        match = pattern.search(filename)
        speaker_ids.append(match.group(1) if match else filename)
    
    raw_dataset = raw_dataset.add_column("speaker_id", speaker_ids)
    
    
    df = raw_dataset.to_pandas()
    df = df.sort_values(by="speaker_id").reset_index(drop=True)
    
    
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_val_idxs, test_idxs = next(splitter.split(df, groups=df["speaker_id"]))

    df_train_val = df.iloc[train_val_idxs].copy().reset_index(drop=True)
    df_test = df.iloc[test_idxs].copy()
    
    
    splitter_val = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idxs, val_idxs = next(splitter_val.split(df_train_val, groups=df_train_val["speaker_id"]))
    
    print(f"Final Split :: Train: {len(train_idxs)} samples | Val: {len(val_idxs)} | Test: {len(test_idxs)} samples")
    dataset = DatasetDict({
        "train": Dataset.from_pandas(df_train_val.iloc[train_idxs]),
        "val": Dataset.from_pandas(df_train_val.iloc[val_idxs]),
        "test": Dataset.from_pandas(df_test)
    })

    train_spk = set(df_train_val.iloc[train_idxs]["speaker_id"])
    val_spk = set(df_train_val.iloc[val_idxs]["speaker_id"])
    test_spk = set(df_test["speaker_id"])
    
    assert train_spk.isdisjoint(val_spk), "Overlap between Train and Val!"
    assert train_spk.isdisjoint(test_spk), "Overlap between Train and Test!"
    
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    return dataset

GLOBAL_DATASET = load_and_split_data()


@dataclass
class AugmentingDataCollator:
    processor: Any
    model_type: str = "whisper"
    augmentor: Optional[Compose] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor, Any]]]) -> Dict[str, torch.Tensor]:
        audio_arrays = [f["audio"]["array"] for f in features]
        sampling_rate = features[0]["audio"]["sampling_rate"]
        text_list = [f["transcription"] for f in features]

        if self.augmentor is not None:
            augmented_audio = []
            for audio in audio_arrays:
                try:
                    aug_sample = self.augmentor(samples=audio.copy(), sample_rate=sampling_rate)
                    augmented_audio.append(aug_sample)
                except Exception as e:
                    print(f"Augmentation Warning: {e}")
                    augmented_audio.append(audio)
            audio_arrays = augmented_audio

        input_features = self.processor(
            audio_arrays, 
            sampling_rate=sampling_rate, 
            return_tensors="pt", 
            padding="max_length"
        ).input_features

        label_features = self.processor.tokenizer(text_list, return_tensors="pt", padding=True, truncation=True)
        labels = label_features.input_ids.masked_fill(label_features.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        return {"input_features": input_features, "labels": labels}


def calculate_f1(ref_str, pred_str):
    ref_tokens = ref_str.lower().split()
    pred_tokens = pred_str.lower().split()
    if len(ref_tokens) == 0 and len(pred_tokens) == 0: return 1.0
    common = Counter(ref_tokens) & Counter(pred_tokens)
    num_same = sum(common.values())
    if num_same == 0: return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    if (precision + recall) == 0: return 0.0
    return 2 * (precision * recall) / (precision + recall)

def compute_str_metrics(pred_strs, label_strs):
    jiwer_out = jiwer.process_words(label_strs, pred_strs)

    cer = jiwer.cer(label_strs, pred_strs)
    
    exact_matches = [1.0 if p.strip() == l.strip() else 0.0 for p, l in zip(pred_strs, label_strs)]
    f1_scores = [calculate_f1(l, p) for l, p in zip(label_strs, pred_strs)]
    
    return {
        "wer": jiwer_out.wer,
        "mer": jiwer_out.mer,
        "wip": jiwer_out.wip,
        "cer": cer,
        "exact_match": np.mean(exact_matches),
        "f1": np.mean(f1_scores)
    }

def create_compute_metrics(processor):
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        return compute_str_metrics(pred_str, label_str)
    return compute_metrics



def train(config=None):
    with wandb.init(config=config, project=WANDB_PROJECT) as run:
        config = wandb.config
        
        print(f"--- Run: {run.name} | Model: {config.model_id} ---")
        
        
        augmentation_pipeline = Compose([
            OneOf([
                AddGaussianNoise(min_amplitude=0.001, max_amplitude=config.noise_max_amp, p=1.0),
                PitchShift(min_semitones=-config.pitch_shift_max, max_semitones=config.pitch_shift_max, p=1.0),
            ], p=1.0)
        ], p=config.aug_prob)

        
        use_4bit = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", 
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16
        ) if use_4bit else None

        processor = WhisperProcessor.from_pretrained(config.model_id, language="English", task="transcribe")
        model = WhisperForConditionalGeneration.from_pretrained(
            config.model_id, quantization_config=bnb_config, device_map="auto"
        )
        
        model.config.use_cache = False 
        model.generation_config.language = "english"
        model.generation_config.task = "transcribe"
        model.generation_config.forced_decoder_ids = None

        if use_4bit: model = prepare_model_for_kbit_training(model)
        peft_config = LoraConfig(
            r=config.lora_rank, lora_alpha=config.lora_rank * 2,
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
            lora_dropout=config.dropout, bias="none", task_type=TaskType.SEQ_2_SEQ_LM
        )
        model = get_peft_model(model, peft_config)

        
        train_collator = AugmentingDataCollator(processor=processor, augmentor=augmentation_pipeline)
        eval_collator = AugmentingDataCollator(processor=processor, augmentor=None) 

        
        args = Seq2SeqTrainingArguments(
            output_dir=f"{OUTPUT_DIR_BASE}/{run.name}",
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation,
            learning_rate=config.learning_rate,
            warmup_steps=50,
            num_train_epochs=config.epochs,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            fp16=False, bf16=True,
            eval_strategy="steps", eval_steps=100, logging_steps=10, save_steps=100,
            predict_with_generate=True, generation_max_length=400,
            load_best_model_at_end=True, metric_for_best_model="wip", greater_is_better=True,
            report_to="wandb", remove_unused_columns=False
        )

        
        class AugmentedTrainer(Seq2SeqTrainer):
            def get_eval_dataloader(self, eval_dataset=None):
                
                return torch.utils.data.DataLoader(
                    eval_dataset if eval_dataset is not None else self.eval_dataset,
                    batch_size=self.args.eval_batch_size,
                    collate_fn=eval_collator, 
                    num_workers=self.args.dataloader_num_workers,
                    pin_memory=self.args.dataloader_pin_memory,
                )

            def get_test_dataloader(self, test_dataset):
                
                return torch.utils.data.DataLoader(
                    test_dataset,
                    batch_size=self.args.eval_batch_size,
                    collate_fn=eval_collator,
                    num_workers=self.args.dataloader_num_workers,
                    pin_memory=self.args.dataloader_pin_memory,
                )

        trainer = AugmentedTrainer(
            model=model, args=args,
            train_dataset=GLOBAL_DATASET["train"],
            eval_dataset=GLOBAL_DATASET["val"],
            data_collator=train_collator,
            compute_metrics=create_compute_metrics(processor), 
            callbacks=[EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)]
        )

        progress_callback = WandBPredictionProgressCallback(trainer, processor.tokenizer, GLOBAL_DATASET["val"])
        trainer.add_callback(progress_callback)

        
        trainer.train()

        final_eval_metrics = trainer.evaluate()

        
        
        wandb.log({f"final_eval/{k}": v for k, v in final_eval_metrics.items()})

        
        print("\n--- Running Final Test Evaluation ---")
        
        
        
        pred_results = trainer.predict(GLOBAL_DATASET["test"])
        
        
        pred_ids = pred_results.predictions
        label_ids = pred_results.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        
        
        
        test_genders = GLOBAL_DATASET["test"]["gender"]
        
        
        global_metrics = compute_str_metrics(pred_str, label_str)
        wandb.log({f"test_global/{k}": v for k, v in global_metrics.items()})
        print(f"Global Test WER: {global_metrics['wer']:.4f}")

        
        df_results = pd.DataFrame({
            "pred": pred_str,
            "label": label_str,
            "gender": test_genders
        })

        for gender in ["M", "F"]:
            subset = df_results[df_results["gender"] == gender]
            if len(subset) > 0:
                gender_metrics = compute_str_metrics(
                    subset["pred"].tolist(), 
                    subset["label"].tolist()
                )
                
                wandb.log({f"test_{gender}/{k}": v for k, v in gender_metrics.items()})
                print(f"Test ({gender}) WER: {gender_metrics['wer']:.4f}")
            else:
                print(f"Warning: No samples found for gender {gender}")

        
        del model, trainer, processor
        torch.cuda.empty_cache()


sweep_configuration = {
    "method": "bayes", 
    "metric": {"name": "eval/wip", "goal": "maximize"},
    "parameters": {
        "model_id": { "values": ["openai/whisper-large-v2"]}, 
        "learning_rate": {"max": 5e-4, "min": 1e-5},
        "lora_rank": {"values": [64, 96, 128]},
        "dropout": {"values": [0.0, 0.1, 0.15, 0.2, 0.3]},
        "batch_size": {"value": 32},
        "gradient_accumulation": {"value": 4},
        "epochs": {"value": 20},
        "early_stopping_patience": {"value": 6},
        "aug_prob": {"values": [0.4, 0.5, 0.6, 0.7, 0.8]},
        "noise_max_amp": {"values": [0.025, 0.035, 0.04, 0.05]},
        "pitch_shift_max": {"values": [4, 6, 8, 10]}
    }
}

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_configuration, project=WANDB_PROJECT)
    wandb.agent(sweep_id, function=train, count=20)
