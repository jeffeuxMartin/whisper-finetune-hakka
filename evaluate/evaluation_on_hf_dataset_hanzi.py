import argparse
import os
import re
from pathlib import Path
from typing import Union

import torch
from datasets import (
    Audio,
    get_dataset_config_names,
    get_dataset_split_names,
    load_dataset,
)
from jiwer import process_characters
from jiwer.alignment import _construct_comparison_string
from jiwer.process import CharacterOutput, WordOutput
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

import evaluate

cer_metric = evaluate.load("cer")


def visualize_alignment(
    output: Union[WordOutput, CharacterOutput],
    show_measures: bool = True,
    skip_correct: bool = True,
    ids: list[str] = None,
) -> str:
    """
    Visualize the output of [jiwer.process_words][process.process_words] and
    [jiwer.process_characters][process.process_characters]. The visualization
    shows the alignment between each processed reference and hypothesis pair.
    If `show_measures=True`, the output string will also contain all measures in the
    output.

    Args:
        output: The processed output of reference and hypothesis pair(s).
        show_measures: If enabled, the visualization will include measures like the WER
                       or CER
        skip_correct: If enabled, the visualization will exclude correct reference and hypothesis pairs

    Returns:
        (str): The visualization as a string

    Example:
        This code snippet
        ```python
        import jiwer

        out = jiwer.process_words(
            ["short one here", "quite a bit of longer sentence"],
            ["shoe order one", "quite bit of an even longest sentence here"],
        )

        print(jiwer.visualize_alignment(out))
        ```
        will produce this visualization:
        ```txt
        sentence 1
        REF:    # short one here
        HYP: shoe order one    *
                I     S        D

        sentence 2
        REF: quite a bit of  #    #  longer sentence    #
        HYP: quite * bit of an even longest sentence here
                   D         I    I       S             I

        number of sentences: 2
        substitutions=2 deletions=2 insertions=4 hits=5

        mer=61.54%
        wil=74.75%
        wip=25.25%
        wer=88.89%
        ```

        When `show_measures=False`, only the alignment will be printed:

        ```txt
        sentence 1
        REF:    # short one here
        HYP: shoe order one    *
                I     S        D

        sentence 2
        REF: quite a bit of  #    #  longer sentence    #
        HYP: quite * bit of an even longest sentence here
                   D         I    I       S             I
        ```
    """
    references = output.references
    hypothesis = output.hypotheses
    alignment = output.alignments
    is_cer = isinstance(output, CharacterOutput)

    final_str = ""
    for gt, hp, chunks, id in zip(references, hypothesis, alignment, ids):
        if skip_correct and len(chunks) == 1 and chunks[0].type == "equal":
            continue

        final_str += f"sentence {id.replace('.wav', '')}\n"
        final_str += _construct_comparison_string(
            gt, hp, chunks, include_space_seperator=not is_cer
        )
        final_str += "\n"

    if show_measures:
        final_str += f"number of sentences: {len(alignment)}\n"
        final_str += f"substitutions={output.substitutions} "
        final_str += f"deletions={output.deletions} "
        final_str += f"insertions={output.insertions} "
        final_str += f"hits={output.hits}\n"

        if is_cer:
            final_str += f"\ncer={output.cer * 100:.2f}%\n"
        else:
            final_str += f"\nmer={output.mer * 100:.2f}%"
            final_str += f"\nwil={output.wil * 100:.2f}%"
            final_str += f"\nwip={output.wip * 100:.2f}%"
            final_str += f"\nwer={output.wer * 100:.2f}%\n"
    else:
        # remove last newline
        final_str = final_str[:-1]

    return final_str


def is_target_text_in_range(ref):
    if ref.strip() == "ignore time segment in scoring":
        return False
    else:
        return ref.strip() != ""


def get_text(sample):
    if "text" in sample:
        return sample["text"]
    elif "sentence" in sample:
        return sample["sentence"]
    elif "normalized_text" in sample:
        return sample["normalized_text"]
    elif "transcript" in sample:
        return sample["transcript"]
    elif "transcription" in sample:
        return sample["transcription"]
    elif "hanzi" in sample:
        return sample["hanzi"]
    else:
        raise ValueError(
            "Expected transcript column of either 'text', 'sentence', 'normalized_text' or 'transcript'. Got sample of "
            ".join{sample.keys()}. Ensure a text column name is present in the dataset."
        )


def get_text_column_names(column_names):
    if "text" in column_names:
        return "text"
    elif "sentence" in column_names:
        return "sentence"
    elif "normalized_text" in column_names:
        return "normalized_text"
    elif "transcript" in column_names:
        return "transcript"
    elif "transcription" in column_names:
        return "transcription"
    elif "hanzi" in column_names:
        return "hanzi"


whisper_norm = BasicTextNormalizer()


def normalize(batch):
    batch["norm_text"] = get_text(batch)
    batch["norm_text"] = re.sub(r"[\.\?\!\;\,\，\。\？\！]", "", batch["norm_text"])
    batch["norm_text"] = re.sub(r"\s+", "", batch["norm_text"]).strip().lower()
    return batch


def main(args):
    if args.is_public_repo == False:
        os.system(f"mkdir -p {args.temp_ckpt_folder}")
        ckpt_dir_parent = str(Path(args.ckpt_dir).parent)
        os.system(
            f"cp {ckpt_dir_parent}/added_tokens.json {ckpt_dir_parent}/normalizer.json \
        {ckpt_dir_parent}/preprocessor_config.json {ckpt_dir_parent}/special_tokens_map.json \
        {ckpt_dir_parent}/tokenizer_config.json {ckpt_dir_parent}/merges.txt \
        {ckpt_dir_parent}/vocab.json {args.ckpt_dir}/config.json \
        {args.ckpt_dir}/model-00001-of-00002.safetensors {args.ckpt_dir}/model-00002-of-00002.safetensors {args.ckpt_dir}/model.safetensors.index.json \
        {args.ckpt_dir}/generation_config.json \
        {args.ckpt_dir}/training_args.bin {args.temp_ckpt_folder}"
        )
        model_id = args.temp_ckpt_folder
    else:
        model_id = args.hf_model

    processor = WhisperProcessor.from_pretrained(
        model_id, language=args.language, task="transcribe"
    )
    # if you not add this line, tokenizer will not add language token and task token to the beginning of the sentence
    # transformers v5 my be fixed
    processor.tokenizer.set_prefix_tokens(language=args.language, task="transcribe")

    model = WhisperForConditionalGeneration.from_pretrained(
        model_id,
        attn_implementation="sdpa",
    ).to("cuda")

    # for transformers > 4.50
    if hasattr(model.config, "forced_decoder_ids"):
        model.config.forced_decoder_ids = None

        if hasattr(model, "generation_config"):
            if hasattr(model.generation_config, "forced_decoder_ids"):
                model.generation_config.forced_decoder_ids = None

    result = {}

    for config in get_dataset_config_names(args.dataset):
        if "test" not in get_dataset_split_names(args.dataset, config):
            continue
        dataset = load_dataset(
            args.dataset,
            config,
            split="test",
        )

        text_column_name = get_text_column_names(dataset.column_names)
        dataset = dataset.filter(lambda x: x["duration"] < 30.0)
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
        dataset = dataset.map(normalize, num_proc=4)
        # dataset = dataset.filter(
        #     is_target_text_in_range, input_columns=[text_column_name], num_proc=2
        # )

        predictions = []
        references = []
        norm_predictions = []
        norm_references = []

        def map_to_pred(batch):
            audios = batch["audio"]
            input_features = processor(
                [audio["array"] for audio in audios],
                sampling_rate=audios[0]["sampling_rate"],
                return_tensors="pt",
            ).input_features

            references.extend(batch[text_column_name])
            norm_references.extend(batch["norm_text"])

            with torch.no_grad():
                predicted_ids = model.generate(
                    input_features.to("cuda"),
                    language=args.language,
                    task="transcribe",
                )
            transcriptions = processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )
            predictions.extend(transcriptions)
            for transcription in transcriptions:
                norm_prediction = transcription
                norm_prediction = re.sub(
                    r"[\.\?\!\;\,\，\。\？\！]", "", norm_prediction
                )
                norm_prediction = re.sub(r"\s+", "", norm_prediction).strip().lower()
                norm_predictions.append(norm_prediction)
            return {}

        dataset.map(map_to_pred, batched=True, batch_size=args.batch_size)

        cer = cer_metric.compute(references=references, predictions=predictions)
        cer = round(100 * cer, 2)

        norm_cer = cer_metric.compute(
            references=norm_references, predictions=norm_predictions
        )
        norm_cer = round(100 * norm_cer, 2)

        print("CER : ", cer)
        print("NORMALIZED CER : ", norm_cer)

        os.system(f"mkdir -p {args.output_dir}")
        dset = args.dataset.replace("/", "_") + "_" + config + "_hanzi"
        op_file = args.output_dir + "/" + dset
        if args.is_public_repo:
            op_file = op_file + "_" + args.hf_model.replace("/", "_")
        else:
            op_file = op_file + "_" + args.ckpt_dir.split("/")[-1].replace("/", "_")
        result_file = open(op_file, "w")
        result_file.write("CER: " + str(cer) + "\n")
        result_file.write("NORMALIZED CER: " + str(norm_cer) + "\n\n\n")

        if args.log_normalized:
            out = process_characters(norm_references, norm_predictions)
            result_file.write(
                visualize_alignment(
                    out,
                    show_measures=True,
                    skip_correct=False,
                    ids=dataset["id"],
                )
            )
        else:
            out = process_characters(references, predictions)
            result_file.write(
                visualize_alignment(
                    out,
                    show_measures=True,
                    skip_correct=False,
                    ids=dataset["id"],
                )
            )
        result_file.close()

    if args.is_public_repo == False:
        os.system(f"rm -r {args.temp_ckpt_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--is_public_repo",
        required=False,
        default=True,
        type=lambda x: (str(x).lower() == "true"),
        help="If the model is available for download on huggingface.",
    )
    parser.add_argument(
        "--hf_model",
        type=str,
        required=False,
        default="openai/whisper-tiny",
        help="Huggingface model name. Example: openai/whisper-tiny",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        required=False,
        default=".",
        help="Folder with the pytorch_model.bin file",
    )
    parser.add_argument(
        "--temp_ckpt_folder",
        type=str,
        required=False,
        default="temp_dir",
        help="Path to create a temporary folder containing the model and related files needed for inference",
    )
    parser.add_argument(
        "--language",
        type=str,
        required=False,
        default="hi",
        help="Two letter language code for the transcription language, e.g. use 'hi' for Hindi. This helps initialize the tokenizer.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        default="mozilla-foundation/common_voice_11_0",
        help="Dataset from huggingface to evaluate the model on. Example: mozilla-foundation/common_voice_11_0",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=16,
        help="Number of samples to go through each streamed batch.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        default="predictions_dir",
        help="Output directory for the predictions and hypotheses generated.",
    )

    parser.add_argument(
        "--log_normalized",
        required=False,
        default=False,
        type=lambda x: (str(x).lower() == "true"),
        help="If True, the normalized ref and hyp are logged.",
    )

    args = parser.parse_args()
    main(args)
