import argparse
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch
from datasets import (
    Audio,
    DatasetDict,
    concatenate_datasets,
    get_dataset_config_names,
    get_dataset_split_names,
    load_dataset,
)
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

import evaluate

#######################     ARGUMENT PARSING        #########################

parser = argparse.ArgumentParser(
    description="Fine-tuning script for Whisper Models of various sizes."
)
parser.add_argument(
    "--model_name",
    type=str,
    required=False,
    default="openai/whisper-small",
    help="Huggingface model name to fine-tune. Eg: openai/whisper-small",
)
parser.add_argument(
    "--language",
    type=str,
    required=False,
    default="Hindi",
    help="Language the model is being adapted to in Camel case.",
)
parser.add_argument(
    "--sampling_rate",
    type=int,
    required=False,
    default=16000,
    help="Sampling rate of audios.",
)
parser.add_argument(
    "--num_proc",
    type=int,
    required=False,
    default=2,
    help="Number of parallel jobs to run. Helps parallelize the dataset prep stage.",
)
parser.add_argument(
    "--train_strategy",
    type=str,
    required=False,
    default="steps",
    help="Training strategy. Choose between steps and epoch.",
)
parser.add_argument(
    "--learning_rate",
    type=float,
    required=False,
    default=1.75e-5,
    help="Learning rate for the fine-tuning process.",
)
parser.add_argument(
    "--warmup_steps",
    type=int,
    required=False,
    default=0,
    help="Number of warmup steps.",
)
parser.add_argument(
    "--warmup_ratio", type=float, required=False, default=0.0, help="Warmup ratio."
)
parser.add_argument(
    "--train_batchsize",
    type=int,
    required=False,
    default=48,
    help="Batch size during the training phase.",
)
parser.add_argument(
    "--eval_batchsize",
    type=int,
    required=False,
    default=32,
    help="Batch size during the evaluation phase.",
)
parser.add_argument(
    "--num_epochs",
    type=int,
    required=False,
    default=20,
    help="Number of epochs to train for.",
)
parser.add_argument(
    "--num_steps",
    type=int,
    required=False,
    default=100000,
    help="Number of steps to train for.",
)
parser.add_argument(
    "--resume_from_ckpt",
    type=str,
    required=False,
    default=None,
    help="Path to a trained checkpoint to resume training from.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    required=False,
    default="output_model_dir",
    help="Output directory for the checkpoints generated.",
)
parser.add_argument(
    "--train_datasets",
    type=str,
    nargs="+",
    required=True,
    default=[],
    help="List of datasets to be used for training.",
)

parser.add_argument(
    "--eval_datasets",
    type=str,
    nargs="+",
    required=True,
    default=[],
    help="List of datasets to be used for evaluation.",
)

parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    required=False,
    default=1,
    help="Number of gradient accumulation steps.",
)

args = parser.parse_args()

if args.train_strategy not in ["steps", "epoch"]:
    raise ValueError("The train strategy should be either steps and epoch.")

if len(args.train_datasets) == 0:
    raise ValueError("No train dataset has been passed")
if len(args.eval_datasets) == 0:
    raise ValueError("No evaluation dataset has been passed")

print("\n\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n")
print("ARGUMENTS OF INTEREST:")
print(vars(args))
print("\n\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n")

gradient_checkpointing = False
freeze_feature_encoder = False
freeze_encoder = False

do_normalize_eval = True
do_lower_case = False
do_remove_punctuation = False
normalizer = BasicTextNormalizer()


#############################       MODEL LOADING       #####################################
processor = WhisperProcessor.from_pretrained(
    args.model_name, language=args.language, task="transcribe"
)

# if you not add this line, tokenizer will not add language token and task token to the beginning of the sentence
# transformers v5 my be fixed
processor.tokenizer.set_prefix_tokens(language=args.language, task="transcribe")

model = WhisperForConditionalGeneration.from_pretrained(
    args.model_name,
    attn_implementation="sdpa",
)


if model.config.decoder_start_token_id is None:
    raise ValueError(
        "Make sure that config.decoder_start_token_id is correctly defined"
    )

if freeze_feature_encoder:
    model.freeze_feature_encoder()

if freeze_encoder:
    model.freeze_encoder()
    model.model.encoder.gradient_checkpointing = False


model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

if gradient_checkpointing:
    model.config.use_cache = False


def remove_trailing_comma_and_add_dot(x):
    sentence = re.sub(r"，$", "", x["sentence"])
    if sentence[-1] not in ["。", "？", "！"]:
        sentence += "。"
    return {"sentence": sentence}


############################        DATASET LOADING AND PREP        ##########################


def load_all_datasets(split):
    combined_dataset = []
    if split == "train":
        for i, ds in enumerate(args.train_datasets):
            for config_name in get_dataset_config_names(ds):
                dataset = load_dataset(
                    ds,
                    config_name,
                    split="train",
                )

                none_audio_dataset = dataset.filter(lambda x: x["audio"] is None)
                if len(none_audio_dataset) > 0:
                    print(
                        f"Dataset {ds} with config {config_name} has {len(none_audio_dataset)} samples with no audio."
                    )
                    for sample in none_audio_dataset:
                        print(f"id {sample['id']} no audio")
                    dataset = dataset.filter(lambda x: x["audio"] is not None)

                dataset = dataset.cast_column("audio", Audio(args.sampling_rate))
                dataset = dataset.rename_column("hanzi", "sentence")
                dataset = dataset.remove_columns(
                    set(dataset.features.keys()) - set(["audio", "sentence"])
                )
                dataset = dataset.map(remove_trailing_comma_and_add_dot, num_proc=8)
                combined_dataset.append(dataset)
    elif split == "eval":
        for i, ds in enumerate(args.eval_datasets):
            for config_name in get_dataset_config_names(ds):
                if "test" not in get_dataset_split_names(ds, config_name):
                    continue
                dataset = load_dataset(ds, config_name, split="test")
                dataset = dataset.cast_column("audio", Audio(args.sampling_rate))
                dataset = dataset.rename_column("hanzi", "sentence")
                dataset = dataset.remove_columns(
                    set(dataset.features.keys()) - set(["audio", "sentence"])
                )

                dataset = dataset.map(remove_trailing_comma_and_add_dot, num_proc=8)
                combined_dataset.append(dataset)

    ds_to_return = concatenate_datasets(combined_dataset)
    ds_to_return = ds_to_return.shuffle(seed=22)
    return ds_to_return


max_label_length = model.config.max_length
min_input_length = 3.0
max_input_length = 30.0


def is_in_length_range(audio, sentence):
    input_length = len(audio["array"]) / audio["sampling_rate"]
    label_length = len(processor.tokenizer.tokenize(sentence))
    return (
        min_input_length < input_length < max_input_length
        and 0 < label_length < max_label_length
    )


print("DATASET PREPARATION IN PROGRESS...")
raw_dataset = DatasetDict()
raw_dataset["train"] = load_all_datasets("train")
raw_dataset["eval"] = load_all_datasets("eval")

raw_dataset = raw_dataset.filter(
    is_in_length_range,
    input_columns=["audio", "sentence"],
    num_proc=args.num_proc,
)

###############################     DATA COLLATOR AND METRIC DEFINITION     ########################


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # 1) Build batch input_features from raw audio
        audios = [f["audio"] for f in features]
        arrays = [a["array"] for a in audios]
        sampling_rate = audios[0]["sampling_rate"] if len(audios) > 0 else None
        fe_out = self.processor.feature_extractor(
            arrays, sampling_rate=sampling_rate, return_tensors="pt"
        )
        batch: Dict[str, torch.Tensor] = {
            "input_features": fe_out["input_features"],
        }

        # 2) Prepare labels by tokenizing transcripts
        sentences: List[str] = [f["sentence"] for f in features]
        if do_lower_case:
            sentences = [s.lower() for s in sentences]
        if do_remove_punctuation and normalizer is not None:
            sentences = [normalizer(s).strip() for s in sentences]

        # Remove spaces from sentences if needed
        sentences = [s.replace(" ", "") for s in sentences]

        labels_batch = self.processor.tokenizer(
            sentences, padding=True, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Remove BOS token if present (added automatically by tokenizer)
        bos_id = self.processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        if labels.size(1) > 0 and (labels[:, 0] == bos_id).all().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
)
print("DATASET PREPARATION COMPLETED")


metric = evaluate.load("cer")


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    if do_normalize_eval:
        pred_str = [normalizer(pred) for pred in pred_str]
        label_str = [normalizer(label) for label in label_str]
        print(pred_str[0])
        print(label_str[0])

    cer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"cer": cer}


###############################     TRAINING ARGS AND TRAINING      ############################

if args.train_strategy == "epoch":
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batchsize,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        # warmup_steps=args.warmup_steps,
        # warmup_ratio=args.warmup_ratio,
        gradient_checkpointing=gradient_checkpointing,
        bf16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=args.num_epochs,
        save_total_limit=10,
        per_device_eval_batch_size=args.eval_batchsize,
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=500,
        report_to=["wandb"],
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        optim="schedule_free_radam",
        lr_scheduler_type="constant",
        # optim="apollo_adamw",
        # optim_target_modules=[r".*.attn.*", r".*.mlp.*"],
        resume_from_checkpoint=args.resume_from_ckpt,
        dataloader_num_workers=16,
        max_grad_norm=1.0,
        remove_unused_columns=False,
    )

elif args.train_strategy == "steps":
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batchsize,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        # warmup_steps=args.warmup_steps,
        # warmup_ratio=args.warmup_ratio,
        gradient_checkpointing=gradient_checkpointing,
        bf16=True,
        eval_strategy="steps",
        eval_steps=800,
        save_strategy="steps",
        save_steps=800,
        max_steps=args.num_steps,
        save_total_limit=10,
        per_device_eval_batch_size=args.eval_batchsize,
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=500,
        report_to=["wandb"],
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        optim="schedule_free_adamw",
        lr_scheduler_type="constant",
        resume_from_checkpoint=args.resume_from_ckpt,
        dataloader_num_workers=16,
    )

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=raw_dataset["train"],
    eval_dataset=raw_dataset["eval"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

processor.save_pretrained(training_args.output_dir)

print("TRAINING IN PROGRESS...")
trainer.train()
print("DONE TRAINING")
print("DONE TRAINING")
print("DONE TRAINING")
