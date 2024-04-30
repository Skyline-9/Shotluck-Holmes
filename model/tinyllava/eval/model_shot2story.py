import argparse
import torch

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

def eval_model(args):
    # Model
    disable_torch_init()

    model_path = args.model_path

    if args.model_path == "NULL":
        model_path = "bczhou/TinyLLaVA-3.1B"
        

    model_name = get_model_name_from_path(model_path)
    print(model_name)
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


    video_object =  make_supervised_data_module(tokenizer, args)['train_dataset']

    for i in range(0, 20000, 50):
        index = i
        video_tensor = video_object[index]

        input_ids = video_tensor['input_ids'].unsqueeze(0).cuda()


        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]

        with torch.inference_mode():
            output_ids = model.generate(
                        input_ids,
                        images=video_tensor['image'].unsqueeze(0).half().cuda(),
                        do_sample=False,
                        max_new_tokens=128,
                        # no_repeat_ngram_size=3,
                        use_cache=True)

            outputs = tokenizer.batch_decode(
                output_ids, skip_special_tokens=True
            )[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[: -len(stop_str)]
            outputs = outputs.strip()
        
            file_path = os.path.join(args.output_dir, "outputs.txt")

            ground_truth = "Ground Truth: " + str(video_object.get_ground_truth(index))
            our_model = "Shotluck Holmes: " + str(outputs)
    


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
