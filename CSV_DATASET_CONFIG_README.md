# whisper-finetune-hakka — CSV dataset manifest (fork patch v2)

Target fork: https://github.com/jeffeuxMartin/whisper-finetune-hakka

Upstream base: `train/fine-tune_on_hf_dataset_all_config.py`, Git blob
`6dd3efb930e58a80f162aaeaccfd6b8197d37e81`.
**Apply this patch to the clean original fork, not on top of an earlier CSV or mini8 patch.**

## Contents

- `whisper_fork_csv_v2.patch`: a single git-apply patch against the original training Python
- `datasets.colab.csv`: HF datasets, `head,25`, local data disabled
- `datasets.nano4.csv`: same HF datasets with `all`, plus local CSV enabled
- `CSV_DATASET_CONFIG_README.md`: these instructions

The script continues using the original Whisper, collator, CER, optimizer, epoch/steps
strategies, and Seq2SeqTrainer. All train/eval sources are declared in one CSV,
selected by `role`; one loader normalizes records to `audio` + `sentence`.

## Apply to the fork and commit

```bash
git clone https://github.com/jeffeuxMartin/whisper-finetune-hakka.git
cd whisper-finetune-hakka
# Copy all four extracted files here.
F=train/fine-tune_on_hf_dataset_all_config.py
cp "$F" "${F%.py}.0.py"    # optional local backup; do NOT commit .0.py

git apply --check whisper_fork_csv_v2.patch
git apply whisper_fork_csv_v2.patch
python -m py_compile "$F"
vimdiff "$F" "${F%.py}.0.py"

git add "$F" datasets.colab.csv datasets.nano4.csv CSV_DATASET_CONFIG_README.md
git diff --cached --stat
git commit -m "Refactor dataset loading to CSV manifest with sampling controls"
git push origin main
```

The backup `.0.py` stays untracked and should not be committed. A GitHub-authenticated
terminal is required for push; never paste access tokens into the notebook output.

## CSV schema

The CSV is one table for train and eval, and has three groups of fields.

**Common fields:**

| Column | Meaning |
|---|---|
| `role` | `train` or `eval` (these filter rows before loading) |
| `source_type` | `hf` or `local_csv` |
| `path` | HF repository (e.g. `openslr/librispeech_asr`) or local index CSV path |
| `config` | HF dataset configuration, e.g. `clean`, `Hakka_Sixian`; blank for local CSV |
| `split` | Exact HF split, e.g. `train.100`, `validation`; or local CSV split value, e.g. `Train` |
| `text_col` | Existing transcript field (`hanzi`, `text`, `transcription`); blank when using a local transcript file |
| `sample_mode` | `all`, `head`, or `ratio` |
| `sample_value` | Blank for `all`; integer (e.g. `25`) for `head`; fraction (e.g. `0.001`) for `ratio` |
| `enabled` | `1` to include the row, `0` to exclude it |

**Optional HF field:**

- `audio_col`: audio column name, default `audio` when blank.

**Local CSV only:**

- `split_col`: index CSV column identifying split, e.g. `Split_Set`.
- `file_col`: index CSV filename/ID column, e.g. `fileName`.
- `audio_template`: relative audio path using `{stem}` or `{name}`, e.g. `Audios/{stem}.wav`.
- `text_template`: relative UTF-8 transcript file path, e.g. `transcripts/{stem}.txt`.
- If local CSV directly includes audio paths and transcripts, set `audio_col` and
  `text_col` to the corresponding columns and leave templates blank.

`{stem}` removes the filename extension; `{name}` retains it. Local paths are relative
to the index CSV location. `$USER` and `~` expand in local index path. The provided
BIIC rows are **ordinary local_csv mappings**; neither BIIC nor its file names are
special-cased in Python.

**Sampling:**

| sample_mode | sample_value | Behavior |
|---|---|---|
| `head` | `25` | Take first 25 samples (HF streaming, no eager whole-dataset download) |
| `ratio` | `0.001` | Metadata row count × 0.001, ceil and min 3 samples, capped at split length |
| `all` | empty | Use the complete split (HF normally loads full dataset) |

For `ratio`, an HTTP `/size` metadata response is cached once per HF repository
within the Python process. Only missing metadata entries fall back to
`load_dataset_builder`. Each new training process may resolve files once when
loading a stream; this is not necessarily a full download. `head` is preferable
for cheap Colab pipeline checks. `take` retrieves a prefix, **not a random sample**;
Parquet downloads may cover entire row groups.

For local CSV, `ratio` counts the matching split in a first CSV pass, then takes
that many valid file references in a second pass.

## Colab CLI

Run in a GPU runtime after cloning your fork and installing repo dependencies.

```bash
python -u train/fine-tune_on_hf_dataset_all_config.py \
  --dataset_config_csv datasets.colab.csv \
  --model_name openai/whisper-tiny \
  --language zh \
  --sampling_rate 16000 \
  --num_proc 1 \
  --train_strategy epoch \
  --num_epochs 1 \
  --train_batchsize 2 \
  --eval_batchsize 2 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-4 \
  --logging_steps 1 \
  --dataloader_num_workers 0 \
  --report_to none \
  --output_dir /content/whisper-csv-smoke
```

Loss is logged at optimizer steps; actual cell-side live rendering is controlled by
Colab/Jupyter's subprocess handling. No loss can appear until `trainer.train()`.

## NCHC Nano4 CLI (within an allocated Slurm GPU job)

```bash
export LD_LIBRARY_PATH="$HOME/.local/ffmpeg6/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
uv run --no-sync python -u train/fine-tune_on_hf_dataset_all_config.py \
  --dataset_config_csv datasets.nano4.csv \
  --model_name openai/whisper-large-v2 \
  --language zh \
  --sampling_rate 16000 \
  --num_proc 8 \
  --train_strategy epoch \
  --num_epochs 5 \
  --train_batchsize 2 \
  --eval_batchsize 2 \
  --gradient_accumulation_steps 64 \
  --logging_steps 100 \
  --dataloader_num_workers 16 \
  --report_to none \
  --output_dir "/work/$USER/whisper-large-v2"
```

For single-node 4-GPU use, allocate **four GPUs** and substitute `torchrun
--standalone --nproc_per_node=4` for `python -u`. Running this full-data command
on a login node is not supported. A real full run may need substantial storage,
CPU preparation time and GPU credits: do a bounded smoke job first.

## Differences from original

- `--dataset_config_csv` replaces required `--train_datasets/--eval_datasets`.
- New `--logging_steps` (default 500), `--dataloader_num_workers` (default 16),
  `--report_to {wandb,none}` (default wandb).
- Sets `logging_first_step=True` so the first update is visible.
- `--num_proc 1` now means non-multiprocessing (`num_proc=None` internally);
  >1 keeps multiprocessing.
- Auto-selects bf16 when supported, otherwise fp16 on GPU, fp32 on CPU.
- `trainer.train(resume_from_checkpoint=...)` replaces the invalid argument
  to `Seq2SeqTrainingArguments` in the original source.
- Source-independent text punctuation logic is preserved, except empty text no
  longer indexes `sentence[-1]`.
- **Other original behavior remains**: it strips spaces from all transcripts,
  appends Chinese punctuation, uses a single Whisper decoder language prefix,
  and measures CER on mixed languages. This is a pipeline smoke test, **not**
  a scientifically sound multilingual-training setup without further changes.

## Validation status

Tested against exact upstream file Git blob `6dd3ef...`: `git apply --check`,
apply, reverse-check and Python syntax compilation all passed. Unit-style offline
tests exercised local split/path/text adapters, `head`/`ratio`/`all`, HF `/size`
metadata caching for multiple splits in a repo, and sample-limit calculation.

**Not yet verified end-to-end** on actual Colab/Nano4 GPUs or actual BIIC files.
An offline test cannot prove that the external FFmpeg/TorchCodec/Trainer stack
will finish training; preserve logs and test a few steps before full training.
