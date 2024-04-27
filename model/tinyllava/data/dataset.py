import copy
from dataclasses import dataclass
import json
from typing import Dict,  Sequence


import transformers
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile

from tinyllava.arguments import *
from tinyllava.utils import *
from tinyllava.data.process import *
from tinyllava.constants import *

# imports for load_video()
from decord import VideoReader
from decord.ndarray import NDArray
import numpy as np
import os

print(f"HERERE:RJEKL:RE {os.getcwd()}")


ImageFile.LOAD_TRUNCATED_IMAGES = True

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        
        # print(sources)

        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result

                image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        elif 'video' in sources[0]:
            # TODO: load video into tensor
            video_file = self.list_data_dict[i]['video']
            video_folder = self.data_args.image_folder
            
            # image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            # print(video_path)

            video = self.load_video(os.path.join(video_folder, video_file), height=384, width=384)
            video = video.permute(1, 0, 2, 3)

            # print(type(tmp))
            # print(tmp.size())

            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        # print(preprocess)
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif 'video' in self.list_data_dict[i]:
            data_dict['image'] = video
            # data_dict['video'] = video
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict

    # n_frms=MAX_INT
    def load_video(self, video_path, n_frms=100, height=-1, width=-1, sampling="uniform", clips=None):
        # check video_path
        if type(video_path) is not str:
            file_obj = io.BytesIO(video_path)
            vr = VideoReader(file_obj, height=height, width=width)
        else:
            vr = VideoReader(uri=video_path, height=height, width=width)
        total_len = len(vr)

        if clips is not None:
            frms = []
            fps = vr.get_avg_fps()
            for clip in clips:
                strt_senconds, end_seconds, shot_strt_frms, shot_end_frms = clip
                if shot_strt_frms == 0 and shot_end_frms == -1:
                    start = strt_senconds * fps + shot_strt_frms
                    end = end_seconds * fps + shot_strt_frms
                    vlen = end - start
                else:
                    vlen = shot_end_frms - shot_strt_frms
                    start = strt_senconds * fps + shot_strt_frms
                    end = strt_senconds * fps + shot_end_frms
                if end > total_len:
                    print(f'Video starts from {start} to {end} exceeding total len {total_len}')
                    end = total_len
                    vlen = end - start
                if n_frms > vlen:
                    indices = np.arange(start, end, vlen / n_frms).astype(int)
                elif sampling == "uniform":
                    indices = np.arange(start, end, vlen / n_frms).astype(int)
                elif sampling == "headtail":
                    half = n_frms // 2
                    another_half = n_frms - half
                    sampled_cnt = [half, another_half]
                    random.shuffle(sampled_cnt)
                    indices_h = sorted(rnd.sample(range(vlen // 2), sampled_cnt[0]))
                    indices_t = sorted(rnd.sample(range(vlen // 2, vlen), sampled_cnt[1]))
                    indices = indices_h + indices_t
                else:
                    raise NotImplementedError
                frms.append(vr.get_batch(indices).permute(3,0,1,2).float())
        else:
            vlen = len(vr)
            #print('video len', vlen)
            start, end = 0, vlen
            #n_frms = min(n_frms, vlen)
            if n_frms > vlen:
                indices = np.arange(start, end, vlen / n_frms).astype(int)
            elif sampling == "uniform":
                indices = np.arange(start, end, vlen / n_frms).astype(int)
            elif sampling == "headtail":
                half = n_frms // 2
                another_half = n_frms - half
                sampled_cnt = [half, another_half]
                random.shuffle(sampled_cnt)
                indices_h = sorted(rnd.sample(range(vlen // 2), sampled_cnt[0]))
                indices_t = sorted(rnd.sample(range(vlen // 2, vlen), sampled_cnt[1]))
                indices = indices_h + indices_t
            else:
                raise NotImplementedError

            # get_batch -> T, H, W, C
            #print(frms)
            #print(frms.shape)
            indices = [int(i) if int(i) < len(vr) else vlen-1 for i in indices]
            indices = sorted(indices)[:n_frms]
            try:
                frms = vr.get_batch(indices)
                if isinstance(frms, torch.Tensor):
                    frms = frms.permute(3,0,1,2).float() 
                elif isinstance(frms, NDArray):
                    frms = torch.from_numpy(frms.asnumpy()).permute(3,0,1,2).float() 
            except Exception as e:
                print(indices, len(vr), n_frms)
                print(video_path)
                indices = [int(i) if int(i) < len(vr) else rnd.sample(range(vlen),1)[0] for i in indices]
                print(indices)
                print(e)
            assert len(frms[0])==n_frms, f"{frms.shape}, {len(frms)}, {indices}, {vlen}, {n_frms}"
            # frms = torch.from_numpy(vr.get_batch(indices).asnumpy()).permute(3, 0, 1, 2).float()  # (C, T, H, W)

        return frms

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # for instance in instances:
        #     print(instance)
        #     print(type(instance))
        #     print("\n\n")
        #     return
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))

        # print(f"input_ids type {type(input_ids)}")
        # print(f"label type: {type(labels)}")
        # print(input_ids)
        # print(labels)
        # return
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == self.tokenizer.eos_token_id] = -300

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        labels = labels[:, :self.tokenizer.model_max_length]
        # FIXME: This is a hack for handling phi and stablelm, as they have the same eos, pad and unk. We want the model
        # FIXME: to predict the eos in the input ids, but we also use the id of eos to pad sequence, so we use a temp
        # FIXME: eos id first, and convert them back.
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == -300] = self.tokenizer.eos_token_id

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                          data_path=data_args.data_path,
                                          data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)
