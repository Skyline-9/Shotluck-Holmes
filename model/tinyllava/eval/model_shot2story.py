import json
import argparse
import os

from tqdm import tqdm
from decord import VideoReader
from decord.ndarray import NDArray

import transformers
import torch
from torch.utils.data import DataLoader, Dataset
import evaluate
from cidereval import cider

from tinyllava.model.builder import load_pretrained_model
from tinyllava.utils import disable_torch_init
from tinyllava.data.dataset import make_supervised_data_module
from tinyllava.conversation import conv_templates
from tinyllava.utils import *
from tinyllava.model import *
from tinyllava.mm_utils import *
from tinyllava.arguments import *


def create_data_loader(video_data, video_folder, tokenizer, image_processor, num_workers=16):
    print("Creating data loader with {} workers".format(num_workers))
    # dataset = CustomDataset(video_data, video_folder, tokenizer, image_processor)
    dataset = make_supervised_data_module(tokenizer, args)['train_dataset']
    data_loader = DataLoader(dataset, batch_size=1, num_workers=num_workers, shuffle=False)
    return dataset, data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = args.model_path

    if args.model_path == "NULL" or True:
        print("No Model Path Provided")
        model_path = "bczhou/TinyLLaVA-3.1B"

    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # Read in annotations
    with open(args.data_path, 'r') as file:
        video_data = json.load(file)
    dataset, data_loader = create_data_loader(video_data, args.data_path, tokenizer, image_processor,
                                              min(os.cpu_count(), 16))

    # Prepare output directory
    os.makedirs(args.output_dir, exist_ok=True)

    gts = []
    preds = []

    # Load evaluation metrics (not used right now)
    bleu = evaluate.load("bleu")
    meteor = evaluate.load("meteor")
    rouge = evaluate.load("rouge")

    model = model.cuda()
    for i, x in tqdm(enumerate(data_loader), total=len(data_loader)):
        input_ids, labels, video = x.values()

        input_ids = input_ids.cuda(non_blocking=True)
        video = video.half().cuda(non_blocking=True)  # maybe use half precision?? .half().cuda()

        # attention_mask = (input_ids != tokenizer.pad_token_id).long()

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=video,
                # attention_mask=attention_mask,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True if args.temperature > 0 else False,
                max_new_tokens=args.max_new_tokens,
                top_p=args.top_p,
                temperature=args.temperature,
                no_repeat_ngram_size=3,
                # stopping_criteria=[stopping_criteria],
                use_cache=True)

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

            gt = str(dataset.get_ground_truth(i))
            preds.append(outputs)
            gts.append([gt])  # Shot2Story20K only has one reference for each video

            # Change for debugging
            if (False):
                print("\033[93m Outputs: {}\033[00m".format(outputs))
                print("\033[93m Ground Truth: {}\033[00m".format(gt))

                bleu_results = bleu.compute(predictions=preds, references=gts)
                print("\033[96m {}\033[00m".format(bleu_results))

                meteor_results = meteor.compute(predictions=preds, references=gts)
                print("\033[96m {}\033[00m".format(meteor_results))

                rouge_results = rouge.compute(predictions=preds, references=gts)
                print("\033[96m {}\033[00m".format(rouge_results))

                cider_results = cider(predictions=preds, references=gts)
                print("\033[96m {}\033[00m".format(cider_results))

    with open(os.path.join(args.output_dir, "gt.json"), "w") as json_file:
        json.dump(gts, json_file)

    with open(os.path.join(args.output_dir, "pred.json"), "w") as json_file:
        json.dump(preds, json_file)

    # bleu_results = bleu.compute(predictions=preds, references=gts)
    # print("\033[96m {}\033[00m".format(bleu_results))

    # meteor_results = meteor.compute(predictions=preds, references=gts)
    # print("\033[96m {}\033[00m".format(meteor_results))

    # rouge_results = rouge.compute(predictions=preds, references=gts)
    # print("\033[96m {}\033[00m".format(rouge_results))

    # cider_results = cider(predictions=preds, references=gts)
    # print("\033[96m {}\033[00m".format(cider_results))


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
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--is_multimodal", type=str, required=True)
    parser.add_argument("--mm_use_im_start_end", type=str, required=True)
    args = parser.parse_args()

    eval_model(args)
