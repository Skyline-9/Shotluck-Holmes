import argparse
import torch
import evaluate
from tqdm import tqdm

from torch.utils.data import DataLoader

from tinyllava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from tinyllava.conversation import conv_templates, SeparatorStyle
from tinyllava.model.builder import load_pretrained_model
from tinyllava.utils import disable_torch_init
from tinyllava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)

import transformers

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re

from tinyllava.data.dataset import make_supervised_data_module
from tinyllava.arguments import *

import os

# DataLoader
def create_data_loader(dataset, model_config, batch_size=1, num_workers=15):
    assert batch_size == 1, "batch_size must be 1"
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)#, collate_fn=collate_fn)
    return data_loader

def eval_model(args):
    # Model
    disable_torch_init()
    model_path = args.model_path

    if args.model_path == "NULL":
        print("HERE")
        model_path = "bczhou/TinyLLaVA-3.1B"

    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name
    )


    qs = ""
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if 'phi' in model_name.lower() or '3.1b' in model_name.lower():
        conv_mode = "phi"
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode
    
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    train_dataset = make_supervised_data_module(tokenizer, args)['train_dataset']

    gts = []
    preds = []

    data_loader = create_data_loader(train_dataset, model.config)

    bleu = evaluate.load("bleu")
    meteor = evaluate.load("meteor")
    rouge = evaluate.load("rouge")

    for i, x in tqdm(enumerate(data_loader), total=len(data_loader)):
        input_ids, labels, video = x.values()
        input_ids = input_ids.cuda()
        labels = labels.cuda()
        video = video.cuda()

        attention_mask = (input_ids != tokenizer.pad_token_id).long()

        with torch.inference_mode():
            output_ids = model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        pad_token_id=tokenizer.eos_token_id,
                        images=video.half().cuda(),
                        do_sample=False,
                        max_new_tokens=128,
                        # no_repeat_ngram_size=3,
                        use_cache=False)

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            outputs = outputs.strip()


            gt = str(train_dataset.get_ground_truth(i))
            # ground_truth = "Ground Truth: " + str(train_dataset.get_ground_truth(i))
            # our_model = "Shotluck Holmes: " + str(outputs)

            if len(outputs) == 0:
                continue

            print(f"VIDEO TITLE: {train_dataset.get_video_title(i)}")
            print(f"PREDICTION: {outputs}\n")
            print(f"GT: {gt}\n\n")
            preds.append(outputs)
            gts.append([gt])

            bleu_results = bleu.compute(predictions=preds, references=gts)
            print(bleu_results)

            meteor_results = meteor.compute(predictions=preds, references=gts)
            print(meteor_results)

            rouge_results = rouge.compute(predictions=preds, references=gts)
            print(rouge_results)
    
    bleu_results = bleu.compute(predictions=preds, references=gts)
    print(bleu_results)

    meteor_results = meteor.compute(predictions=preds, references=gts)
    print(meteor_results)

    rouge_results = rouge.compute(predictions=preds, references=gts)
    print(rouge_results)

    print(f"{len(predictions)}/{len(data_loader)}")

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--conv_mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.3)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--is_multimodal", type=str, required=True)
    parser.add_argument("--mm_use_im_start_end", type=str, required=True)
    args = parser.parse_args()

    eval_model(args)
