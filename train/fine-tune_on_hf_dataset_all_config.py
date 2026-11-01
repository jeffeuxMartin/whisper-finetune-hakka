import argparse
import csv
import json
import math
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Union
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import torch
from datasets import (
    Audio,
    Dataset,
    DatasetDict,
    Features,
    Value,
    concatenate_datasets,
    load_dataset_builder,
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
    "--dataset_config_csv",
    type=str,
    required=True,
    help="CSV manifest defining train/eval datasets, source adapters and sampling.",
)

parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    required=False,
    default=1,
    help="Number of gradient accumulation steps.",
)
parser.add_argument(
    "--logging_steps", type=int, default=500,
    help="Log training loss every N optimizer steps (e.g., 1 for a Colab smoke test).",
)
parser.add_argument(
    "--dataloader_num_workers", type=int, default=None,
    help="PyTorch DataLoader workers; 0 is useful for Colab (default: 16).",
)
parser.add_argument(
    "--report_to", choices=["wandb", "none"], default="wandb",
    help="Trainer reporting backend (use 'none' for Colab without W&B).",
)

args = parser.parse_args()

if args.train_strategy not in ["steps", "epoch"]:
    raise ValueError("The train strategy should be either steps and epoch.")
if args.num_proc < 1 or args.logging_steps < 1:
    raise ValueError("--num_proc and --logging_steps must be positive integers")
if args.dataloader_num_workers is not None and args.dataloader_num_workers < 0:
    raise ValueError("--dataloader_num_workers must be >= 0")

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
    if sentence and sentence[-1] not in ["。", "？", "！"]:
        sentence += "。"
    return {"sentence": sentence}


############################        DATASET LOADING AND PREP        ##########################


# One CSV controls both roles. HF and local_csv sources normalize to the same
# Audio/sentence schema before the original filtering and Trainer code.
DATASET_FEATURES = Features({"audio": Audio(decode=False), "sentence": Value("string")})
MANIFEST = Path(os.path.expandvars(os.path.expanduser(args.dataset_config_csv))).resolve()

with MANIFEST.open(newline="", encoding="utf-8-sig") as handle:
    DATASET_SETTINGS = list(csv.DictReader(handle))

REQUIRED_COLUMNS = {"role", "source_type", "path", "split", "sample_mode", "sample_value"}
if not DATASET_SETTINGS:
    raise ValueError(f"Empty dataset manifest: {MANIFEST}")
if not REQUIRED_COLUMNS.issubset(DATASET_SETTINGS[0]):
    raise ValueError(f"Missing manifest columns: {sorted(REQUIRED_COLUMNS - set(DATASET_SETTINGS[0]))}")


def _enabled(cfg):
    return (cfg.get("enabled") or "1").strip().lower() not in {"0", "false", "no", "off"}


def _local_path(value, base):
    path = Path(os.path.expandvars(os.path.expanduser(value)))
    return path if path.is_absolute() else base / path


@lru_cache(maxsize=None)
def _hf_size_table(repo):
    """One small metadata request per HF repository, shared across configs/splits.

    Dataset Viewer may report partial coverage; individual rows with a concrete
    num_rows remain usable. Missing entries fall back to DatasetBuilder metadata.
    """
    url = "https://datasets-server.huggingface.co/size?" + urlencode({"dataset": repo})
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        with urlopen(Request(url, headers=headers), timeout=30) as response:
            payload = json.load(response)
        return {
            (item["config"], item["split"]): int(item["num_rows"])
            for item in payload.get("size", {}).get("splits", [])
            if item.get("num_rows") is not None
        }
    except Exception as exc:
        print(f"[metadata] /size unavailable for {repo}: {exc}", flush=True)
        return {}


@lru_cache(maxsize=None)
def _hf_num_rows(repo, config, split):
    # Avoid repeatedly resolving all Parquet shards solely to count examples.
    count = _hf_size_table(repo).get((config, split))
    if count is not None:
        return count
    try:
        builder = load_dataset_builder(repo, config or None)
        split_info = (builder.info.splits or {}).get(split)
        if split_info is not None and split_info.num_examples is not None:
            return int(split_info.num_examples)
    except Exception as exc:
        raise ValueError(
            f"No row-count metadata for {repo}/{config}/{split}. "
            "Set sample_mode=head and sample_value=25 in the CSV."
        ) from exc
    raise ValueError(
        f"No row-count metadata for {repo}/{config}/{split}. "
        "Set sample_mode=head and sample_value=25 in the CSV."
    )


def _sample_limit(cfg, count=None):
    mode = (cfg["sample_mode"] or "all").strip().lower()
    if mode == "all":
        return None
    if mode == "head":
        n = int(cfg["sample_value"])
        if n < 1:
            raise ValueError(f"head sample_value must be positive: {cfg}")
        return n
    if mode == "ratio":
        fraction = float(cfg["sample_value"])
        if not 0 < fraction <= 1:
            raise ValueError(f"ratio sample_value must be within (0, 1]: {cfg}")
        if count is None:
            count = _hf_num_rows(cfg["path"], cfg.get("config") or "", cfg["split"])
        return min(count, max(3, math.ceil(count * fraction)))
    raise ValueError(f"Unknown sample_mode={mode!r}: {cfg}")


def _load_hf(cfg):
    limit = _sample_limit(cfg)
    audio_col = cfg.get("audio_col") or "audio"
    text_col = cfg.get("text_col") or "sentence"
    data = load_dataset(
        cfg["path"], cfg.get("config") or None, split=cfg["split"],
        streaming=limit is not None,
    )
    data = data.cast_column(audio_col, Audio(decode=False))
    if audio_col != "audio":
        data = data.rename_column(audio_col, "audio")
    if text_col != "sentence":
        data = data.rename_column(text_col, "sentence")
    data = data.select_columns(["audio", "sentence"])
    if limit is not None:
        # Streaming avoids the full Hub download. Materialize only the selected
        # records, as the original Seq2SeqTrainer expects an indexed Dataset.
        data = Dataset.from_list(list(data.take(limit)), features=DATASET_FEATURES)
    return data


def _local_rows(cfg, read_content=True):
    """Stream the index CSV; never decode audio while selecting rows."""
    index_path = _local_path(cfg["path"], MANIFEST.parent)
    root = index_path.parent
    split_col = cfg.get("split_col") or ""
    file_col = cfg.get("file_col") or ""
    text_col = cfg.get("text_col") or ""
    audio_col = cfg.get("audio_col") or ""
    audio_template = cfg.get("audio_template") or ""
    text_template = cfg.get("text_template") or ""
    with index_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if split_col and (row.get(split_col) or "").strip().casefold() != cfg["split"].strip().casefold():
                continue
            if not read_content:
                yield None
                continue
            name = Path((row.get(file_col) or "").strip()).name if file_col else ""
            parts = {"name": name, "stem": Path(name).stem}
            if audio_col:
                audio_name = (row.get(audio_col) or "").strip()
            else:
                audio_name = audio_template.format(**parts) if name else ""
            if not audio_name:
                continue
            audio_path = _local_path(audio_name, root)
            if not audio_path.is_file():
                continue
            if text_col:
                sentence = (row.get(text_col) or "").strip()
            else:
                text_name = text_template.format(**parts) if name else ""
                if not text_name:
                    continue
                text_path = _local_path(text_name, root)
                if not text_path.is_file():
                    continue
                sentence = text_path.read_text(encoding="utf-8-sig").strip()
            if sentence:
                yield {"audio": {"path": str(audio_path), "bytes": None}, "sentence": sentence}


def _load_local_csv(cfg):
    mode = cfg["sample_mode"].strip().lower()
    if mode == "ratio":
        # Local CSV split metadata is cheap to count in a first pass.
        count = sum(1 for _ in _local_rows(cfg, read_content=False))
        limit = _sample_limit(cfg, count)
    else:
        limit = _sample_limit(cfg)
    records = []
    for record in _local_rows(cfg):
        records.append(record)
        if limit is not None and len(records) >= limit:
            break
    return Dataset.from_list(records, features=DATASET_FEATURES)


# datasets.map/filter with num_proc=1 still spawns a worker subprocess.
# None explicitly runs in this process and avoids the Colab/Python 3.13
# multiprocessing/BytesIO teardown noise seen during the smoke tests.
DATASET_NUM_PROC = args.num_proc if args.num_proc > 1 else None


def load_one_dataset(cfg):
    source = (cfg["source_type"] or "").strip().lower()
    if source == "hf":
        dataset = _load_hf(cfg)
    elif source == "local_csv":
        dataset = _load_local_csv(cfg)
    else:
        raise ValueError(f"Unknown source_type={source!r}: {cfg}")

    # Shared filtering and conversion. Audio is decoded only here, after taking
    # a small sample if requested; the original duration filter runs below.
    dataset = dataset.filter(
        lambda audio, sentence: audio is not None and bool(sentence and sentence.strip()),
        input_columns=["audio", "sentence"],
        num_proc=DATASET_NUM_PROC,
    )
    if len(dataset) == 0:
        raise ValueError(f"Zero usable rows in {cfg['path']}/{cfg['split']}")
    # Normalize labels before enabling audio decoding/resampling.
    dataset = dataset.map(remove_trailing_comma_and_add_dot, num_proc=DATASET_NUM_PROC)
    dataset = dataset.cast_column("audio", Audio(args.sampling_rate))
    print(f"[{cfg['role']}] {source} {cfg['path']} "
          f"({cfg.get('config')}/{cfg['split']}): {len(dataset)} rows", flush=True)
    return dataset


def load_all_datasets(role):
    selected = [cfg for cfg in DATASET_SETTINGS if _enabled(cfg) and cfg["role"] == role]
    if not selected:
        raise ValueError(f"No enabled datasets for role={role!r} in {MANIFEST}")
    combined = [load_one_dataset(cfg) for cfg in selected]
    return concatenate_datasets(combined).shuffle(seed=22)


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
    num_proc=DATASET_NUM_PROC,
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
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=args.num_epochs,
        save_total_limit=10,
        per_device_eval_batch_size=args.eval_batchsize,
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        report_to=[] if (args.report_to == "none" or os.environ.get("WHISPER_NO_WANDB")) else ["wandb"],
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        optim="schedule_free_radam",
        lr_scheduler_type="constant",
        # optim="apollo_adamw",
        # optim_target_modules=[r".*.attn.*", r".*.mlp.*"],
        dataloader_num_workers=(
            args.dataloader_num_workers if args.dataloader_num_workers is not None
            else int(os.environ.get("WHISPER_WORKERS", "16"))
        ),
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
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        eval_strategy="steps",
        eval_steps=800,
        save_strategy="steps",
        save_steps=800,
        max_steps=args.num_steps,
        save_total_limit=10,
        per_device_eval_batch_size=args.eval_batchsize,
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        report_to=[] if (args.report_to == "none" or os.environ.get("WHISPER_NO_WANDB")) else ["wandb"],
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        optim="schedule_free_adamw",
        lr_scheduler_type="constant",
        dataloader_num_workers=(
            args.dataloader_num_workers if args.dataloader_num_workers is not None
            else int(os.environ.get("WHISPER_WORKERS", "16"))
        ),
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
trainer.train(resume_from_checkpoint=None if args.resume_from_ckpt in (None, "None") else args.resume_from_ckpt)
print("DONE TRAINING")
print("DONE TRAINING")
print("DONE TRAINING")
