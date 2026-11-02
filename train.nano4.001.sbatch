#!/usr/bin/env bash
#SBATCH --job-name=whisper-001
#SBATCH --account=MST114435
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=whisper-001-%j.out
#SBATCH --error=whisper-001-%j.err

set -euo pipefail

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
cd "$HOME/Hakka_exp/whisper-finetune-hakka"
test -f datasets.nano4.001.csv
test -f train/fine-tune_on_hf_dataset_all_config.py

# Keep Hugging Face artifacts/checkpoints in persistent /work space.
export HF_HOME="/work/$USER/huggingface"
export XDG_CACHE_HOME="/work/$USER/.cache"
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "/work/$USER/whisper-hakka-001"
export LD_LIBRARY_PATH="$HOME/.local/ffmpeg6/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export WHISPER_NO_WANDB=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

echo "[job] id=$SLURM_JOB_ID node=$(hostname) start=$(date -Is)"
nvidia-smi

uv run --no-sync python -u train/fine-tune_on_hf_dataset_all_config.py \
  --dataset_config_csv datasets.nano4.001.csv \
  --model_name openai/whisper-large-v2 \
  --language zh \
  --sampling_rate 16000 \
  --num_proc 1 \
  --train_strategy epoch \
  --num_epochs 1 \
  --train_batchsize 2 \
  --eval_batchsize 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --logging_steps 1 \
  --dataloader_num_workers 0 \
  --report_to none \
  --output_dir "/work/$USER/whisper-hakka-001"

echo "[job] finished=$(date -Is)"
