#!/bin/bash

# uv run torchrun --nproc_per_node=1 train/fine-tune_dryrun.py \
torchrun --standalone --nproc_per_node=1 train/fine-tune_dryrun.py \
--model_name openai/whisper-large-v2 \
--language zh \
--train_datasets formospeech/hat_asr_sixian_reading_clean \
--eval_datasets formospeech/hakkaradio_news_clean \
--sampling_rate 16000 \
--num_proc 8 \
--train_strategy epoch \
--learning_rate 1e-4 \
--train_batchsize 2 \
--eval_batchsize 8 \
--num_epochs 5 \
--resume_from_ckpt None \
--output_dir outputs/schedule_free_r_1e-4_large_v2_hanzi \
--gradient_accumulation_steps 64

# --train_datasets formospeech/hat_asr_sixian_reading_clean formospeech/hat_asr_sixian_broadcast_clean formospeech/hat_asr_nansixian_reading_clean formospeech/hat_asr_hailu_reading_clean formospeech/hat_tts_hailu_clean formospeech/hat_tts_sixian_clean formospeech/fsr23_eval_clean formospeech/fsr25_warmup_reading_clean formospeech/fsr25_train_clean formospeech/fsr25_final_clean formospeech/fsr25_warmup_media_clean formospeech/hakka_elearning_example_clean formospeech/hakkatv_hanzawa_clean formospeech/hakka_elearning_yt_clean formospeech/hakkaradio_news_clean \
