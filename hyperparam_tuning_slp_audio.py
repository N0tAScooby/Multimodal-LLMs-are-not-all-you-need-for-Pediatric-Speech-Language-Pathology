import argparse
import datetime
import os
import re
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset, Audio, ClassLabel, concatenate_datasets
import wandb
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    TrainingArguments,
    Trainer,
    set_seed,
    EarlyStoppingCallback

)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.model_selection import train_test_split as sklearn_split
from audiomentations import PitchShift, AddGaussianNoise
from torch import nn
import warnings
import random


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


warnings.filterwarnings("ignore", message="Was asked to gather along dimension 0")


parser = argparse.ArgumentParser(description="Train Audio Classifier (Sweep Ready)")


parser.add_argument("--target_column", type=str, required=True, 
                    choices=["disorder_class", "disorder_type", "disorder_symptom"],
                    help="The column to classify")
parser.add_argument("--model_checkpoint", type=str, default="facebook/wav2vec2-large-xlsr-53", 
                    help="HuggingFace model hub checkpoint")
parser.add_argument("--wandb_project", type=str, default="SLP-Gender-CDA", help="W&B Project name")

parser.add_argument("--exclude_healthy", type=str2bool, default=False, 
                    help="If True, removes 'typically_developing' samples from the dataset")


parser.add_argument("--aug_male", type=str2bool, default=True, help="Apply augmentations to Male samples")
parser.add_argument("--aug_female",type=str2bool, default=True, help="Apply augmentations to Female samples")


parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--learning_rate", type=float, default=3e-5)
parser.add_argument("--gradient_accumulation", type=int, default=2)
parser.add_argument("--oversample_mult", type=int, default=5, help="Max duplication factor for balancing")
parser.add_argument("--pitch_prob", type=float, default=0.5, help="Probability of applying pitch shift (if gender matches)")
parser.add_argument("--noise_prob", type=float, default=0.0, help="Probability of applying Gaussian noise")
parser.add_argument("--pitch_shift_min", type=int, default=6, help="Minimum absolute semitones to shift")
parser.add_argument("--pitch_shift_max", type=int, default=8, help="Maximum absolute semitones to shift")


parser.add_argument("--random_seed", type=int, help="Seed for Run (Init/Shuffle/Val-Split)")
parser.add_argument("--doTest", type=bool, default=True, help="should evaluate on test?")

TEST_SPLIT_SEED = 42 


args = parser.parse_args()

if not args.random_seed:
    args.random_seed = random.randint(0,10000)


if args.random_seed:
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    set_seed(args.random_seed) 
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)


print(f"--- RUN CONFIGURATION ---")
print(f"\n[STANDARD LOSS]\n")
print(f"Test Split Seed (Fixed): {TEST_SPLIT_SEED}")
print(f"Run Seed (Variable):     {args.random_seed}")
print(f"-------------------------")


safe_model_name = args.model_checkpoint.replace("/", "-")


timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

DATASET_NAME = "SAA-Lab/SLPHelmUltraSuitePlus"
MAX_DURATION = 12.0 
OUTPUT_DIR = f"./results_{args.target_column}_{safe_model_name}_seed{args.random_seed}_{timestamp}"


dataset = load_dataset(DATASET_NAME, split="train")

def add_speaker_ids(dataset):
    print("Extracting Speaker IDs...")
    ds_meta = dataset.cast_column("audio", Audio(decode=False))
    paths = [item['audio']['path'] for item in ds_meta]
    pattern = re.compile(r"^(?:test_|train_|val_|audio_)?([a-zA-Z0-9]+)[_\-\.]", re.IGNORECASE)
    speaker_ids = []
    for p in paths:
        filename = os.path.basename(p)
        match = pattern.search(filename)
        speaker_ids.append(match.group(1) if match else filename)
    return dataset.add_column("speaker_id", speaker_ids)

dataset = add_speaker_ids(dataset)

def clean_and_remap_labels(batch):
    label = batch[args.target_column]
    if label is None: return batch
    label = str(label).replace("['", "").replace("']", "").replace("[", "").replace("]", "")
    
    
    if args.target_column == "disorder_type":
        if label in ["addition", "omission"]: label = "articulation"
        elif label == "stuttering": label = "phonological"
    elif args.target_column == "disorder_symptom":
        if label in ['substitution_modified', "phonological"]: label = "stuttering" 
    return {args.target_column: label}

dataset = dataset.map(clean_and_remap_labels)
dataset = dataset.filter(lambda x: x[args.target_column] is not None)


def balance_dataset(ds, target_col, max_multiplier):
    labels = ds[target_col]
    counts = {}
    for l in labels: counts[l] = counts.get(l, 0) + 1
    
    if not counts: return ds
    
    majority_count = max(counts.values())
    dfs = [ds]
    
    print(f"\n[Balancing] Initial Distribution: {counts}")
    print(f"[Balancing] Balancing strategy: Capped at {max_multiplier}x duplicates")

    for label, count in counts.items():
        if count == majority_count: continue
        
        ratio = majority_count / count
        multiplier = min(int(round(ratio)), max_multiplier)
        
        if multiplier > 1:
            print(f"   -> Oversampling '{label}': {count} -> {multiplier}x (Target ~{count*multiplier})")
            minority_ds = ds.filter(lambda x: x[target_col] == label)
            for _ in range(multiplier - 1):
                dfs.append(minority_ds)
                
    if len(dfs) > 1:
        balanced_ds = concatenate_datasets(dfs)
        print(f"[Balancing] Final Size: {len(balanced_ds)}")
        return balanced_ds
    return ds




df = dataset.to_pandas()
unique_speakers = sorted(df["speaker_id"].unique())


train_val_spk, test_spk = sklearn_split(unique_speakers, test_size=0.2, random_state=42)


train_spk, val_spk = sklearn_split(train_val_spk, test_size=0.2, random_state=42)


assert set(train_spk).isdisjoint(set(val_spk)), "Leakage: Train <-> Val"
assert set(train_spk).isdisjoint(set(test_spk)), "Leakage: Train <-> Test"


dataset_train = dataset.filter(lambda x: x["speaker_id"] in train_spk)
dataset_val   = dataset.filter(lambda x: x["speaker_id"] in val_spk)
dataset_test  = dataset.filter(lambda x: x["speaker_id"] in test_spk)



if args.target_column in ["disorder_symptom", "disorder_type"]:
    args.exclude_healthy = True

if args.exclude_healthy:
    print(f"\n[Filter] Purging 'typically_developing' samples from all splits...")
    
    dataset_train = dataset_train.filter(lambda x: str(x[args.target_column]).strip().lower() != "typically_developing")
    dataset_val   = dataset_val.filter(lambda x: str(x[args.target_column]).strip().lower() != "typically_developing")
    dataset_test  = dataset_test.filter(lambda x: str(x[args.target_column]).strip().lower() != "typically_developing")


all_labels = set(dataset_train[args.target_column]) | set(dataset_val[args.target_column]) | set(dataset_test[args.target_column])
unique_labels = sorted(list(all_labels))


print(f"Labels the model will learn: {unique_labels}")


label_feature = ClassLabel(names=unique_labels)
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for i, label in enumerate(unique_labels)}
ALL_LABEL_IDS = list(range(len(unique_labels)))


dataset_train = dataset_train.map(lambda x: {"labels": label_feature.str2int(x[args.target_column])})
dataset_val   = dataset_val.map(lambda x: {"labels": label_feature.str2int(x[args.target_column])})
dataset_test  = dataset_test.map(lambda x: {"labels": label_feature.str2int(x[args.target_column])})

print(f"\n[Split Stats] Seed: {args.random_seed}")
print(f"   Train: {len(dataset_train)} samples")
print(f"   Val:   {len(dataset_val)} samples")
print(f"   Test:  {len(dataset_test)} samples [FIXED SPEAKER SPLIT]")


print(f"Applying Class Balancing logic to {args.target_column} (Train Set Only)...")
dataset_train = balance_dataset(dataset_train, args.target_column, args.oversample_mult)

train_labels = dataset_train["labels"]
class_counts = np.bincount(train_labels, minlength=len(unique_labels))
class_counts = np.maximum(class_counts, 1) 
total_samples = len(train_labels)

computed_weights = total_samples / (len(unique_labels) * class_counts)
CLASS_WEIGHTS_TENSOR = torch.tensor(computed_weights, dtype=torch.float32)

print("\n[Auto-Weights] Calculated Loss Penalties:")
for idx, label in id2label.items():
    print(f"   - {label}: {CLASS_WEIGHTS_TENSOR[idx]:.2f}x")


feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_checkpoint)
target_sr = feature_extractor.sampling_rate

dataset_train = dataset_train.cast_column("audio", Audio(sampling_rate=target_sr))
dataset_val = dataset_val.cast_column("audio", Audio(sampling_rate=target_sr))
dataset_test = dataset_test.cast_column("audio", Audio(sampling_rate=target_sr))

p_min = min(args.pitch_shift_min, args.pitch_shift_max)
p_max = max(args.pitch_shift_min, args.pitch_shift_max)

print(f"[Augmentation] Dynamic Pitch Shift: Abs Range [{p_min}, {p_max}] semitones")


augmenter_pitch_up = PitchShift(min_semitones=p_min, max_semitones=p_max, p=1.0)

augmenter_pitch_down = PitchShift(min_semitones=-p_max, max_semitones=-p_min, p=1.0)

augmenter_noise = AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1.0)

def preprocess_train(batch):
    audio_arrays = [x["array"] for x in batch["audio"]]
    genders = batch["gender"]
    processed = []
    
    for x, g in zip(audio_arrays, genders):
        
        if np.random.rand() < args.pitch_prob:
            if g == 'M' and args.aug_male:
                x = augmenter_pitch_up(samples=x, sample_rate=target_sr)
            elif g == 'F' and args.aug_female:
                x = augmenter_pitch_down(samples=x, sample_rate=target_sr)
        
        
        if args.noise_prob > 0 and np.random.rand() < args.noise_prob:
            x = augmenter_noise(samples=x, sample_rate=target_sr)

        processed.append(x)
        
    inputs = feature_extractor(processed, sampling_rate=target_sr, max_length=int(target_sr * MAX_DURATION), truncation=True, padding=True)
    inputs["labels"] = batch["labels"]
    return inputs

def preprocess_val(batch):
    audio_arrays = [x["array"] for x in batch["audio"]]
    inputs = feature_extractor(audio_arrays, sampling_rate=target_sr, max_length=int(target_sr * MAX_DURATION), truncation=True, padding=True)
    inputs["labels"] = batch["labels"]
    return inputs

dataset_train.set_transform(preprocess_train)
dataset_val.set_transform(preprocess_val)
dataset_test.set_transform(preprocess_val)


class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        weights = CLASS_WEIGHTS_TENSOR.to(labels.device)
        loss_fct = nn.CrossEntropyLoss(weight=weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    preds = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids
    p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc, "f1_macro": f1_mac, "f1_micro": acc,
        "precision_macro": p_mac, "recall_macro": r_mac
    }



try:
    model = AutoModelForAudioClassification.from_pretrained(
        args.model_checkpoint, 
        num_labels=len(unique_labels), 
        label2id=label2id, 
        id2label=id2label, 
        ignore_mismatched_sizes=True
    )
except OSError:
    
    model = AutoModelForAudioClassification.from_pretrained(
        args.model_checkpoint, 
        num_labels=len(unique_labels), 
        label2id=label2id, 
        id2label=id2label, 
        ignore_mismatched_sizes=True,
        trust_remote_code=True
    )

model.freeze_feature_encoder()

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    
    learning_rate=args.learning_rate,
    gradient_accumulation_steps=args.gradient_accumulation,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    fp16=torch.cuda.is_available(),
    remove_unused_columns=False,
    report_to="wandb",
    run_name=f"{args.target_column}_balanced",
    logging_steps=10
)


wandb.init(project=args.wandb_project, config=args)

trainer = WeightedTrainer(
    model=model, 
    args=training_args, 
    train_dataset=dataset_train, 
    eval_dataset=dataset_val, 
    processing_class=feature_extractor, 
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=6)]
)

trainer.train()

final_eval_metrics = trainer.evaluate()



wandb.log({
    "eval/loss": final_eval_metrics["eval_loss"],
    "eval/accuracy": final_eval_metrics["eval_accuracy"],
    "eval/f1_macro": final_eval_metrics["eval_f1_macro"],
    "eval/precision_macro": final_eval_metrics["eval_precision_macro"],
    "eval/recall_macro": final_eval_metrics["eval_recall_macro"],
    
    "best_val_f1_macro": final_eval_metrics["eval_f1_macro"]
})


if args.doTest:
    print("\n--- FINAL STRATIFIED EVALUATION ---")
    test_results = trainer.predict(dataset_test)
    y_true = test_results.label_ids
    y_pred = np.argmax(test_results.predictions, axis=1)
    
    dataset_test.reset_format()
    df_eval = pd.DataFrame({'true': y_true, 'pred': y_pred, 'gender': dataset_test['gender']})
    
    
    print("\nGlobal Classification Report:")
    report_dict = classification_report(y_true, y_pred, labels=ALL_LABEL_IDS, target_names=unique_labels, zero_division=0, output_dict=True)
    report_text = classification_report(y_true, y_pred, labels=ALL_LABEL_IDS, target_names=unique_labels, zero_division=0)
    print(report_text)
    with open(os.path.join(OUTPUT_DIR, "global_report.txt"), "w") as f: f.write(report_text)
    
    wandb.log({
        "test_global/f1_macro": report_dict['macro avg']['f1-score'],
        "test_global/f1_micro": report_dict['accuracy'],
        "test_global/precision_macro": report_dict['macro avg']['precision'],
        "test_global/recall_macro": report_dict['macro avg']['recall'],
        "test_conf_mat_global": wandb.plot.confusion_matrix(
            probs=None, y_true=list(y_true), preds=list(y_pred), class_names=unique_labels
        )
    })
    
    
    for gender in ['M', 'F']:
        gender_df = df_eval[df_eval['gender'] == gender]
        if len(gender_df) > 0:
            
            g_true = list(gender_df['true'])
            g_pred = list(gender_df['pred'])
            
            rep_dict = classification_report(g_true, g_pred, labels=ALL_LABEL_IDS,  target_names=unique_labels, zero_division=0, output_dict=True)
            report_text = classification_report(g_true, g_pred, labels=ALL_LABEL_IDS, target_names=unique_labels, zero_division=0)
            
            print(f"\nResults for Gender: {gender}")
            print(report_text)
            
            wandb.log({
                f"test_{gender}/f1_macro": rep_dict['macro avg']['f1-score'],
                f"test_{gender}/f1_micro": rep_dict['accuracy'],
                f"test_{gender}/precision_macro": rep_dict['macro avg']['precision'],
                f"test_{gender}/recall_macro": rep_dict['macro avg']['recall'],
                f"test_conf_mat_{gender}": wandb.plot.confusion_matrix(
                    probs=None, y_true=g_true, preds=g_pred, class_names=unique_labels
                )
            })
            
            with open(os.path.join(OUTPUT_DIR, f"report_{gender}.txt"), "w") as f: f.write(report_text)

wandb.finish()
