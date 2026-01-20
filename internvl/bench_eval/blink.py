import math
import numpy as np
import torch
import time
import requests
import string
from io import BytesIO
import torchvision.transforms as T
import pandas as pd
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import logging
from datetime import datetime

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    if isinstance(image_file, Image.Image):
        image = image_file.convert("RGB")
    elif image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def load_images(image_files, input_size=448, max_num=12):
    pv_list, num_patches_list = [], []
    for image_file in image_files:
        pv = load_image(image_file)
        pv_list.append(pv)
        num_patches_list.append(pv.size(0))
    pixel_values = torch.cat(pv_list, dim=0)

    return pixel_values, num_patches_list

def main():
    model_path = "InternVL3_5-8B"
    model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            load_in_8bit=False,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            attn_implementation="flash_attention_2",
            device_map="auto").eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

    # subtasks = ['Art_Style', 'Functional_Correspondence', 'Multi-view_Reasoning', 'Relative_Reflectance',
    #     'Visual_Correspondence', 'Counting', 'IQ_Test', 'Object_Localization', 'Semantic_Correspondence', 
    #     'Visual_Similarity', 'Forensic_Detection', 'Jigsaw', 'Relative_Depth', 'Spatial_Relation'
    # ]

    correct, wrong, invalid = 0, 0, 0
    total_time = 0
    oom = 0

    ds = load_dataset("BLINK-Benchmark/BLINK")

    for i, d in enumerate(tqdm(ds['validation'])):
        
        try:
            question = d['prompt']
            options = d['choices']
            answer = d['answer'][1]
            images = [d['image_1'], d['image_2'], d['image_3'], d['image_4']]
            image_list = []
            choice_list = ['A', 'B', 'C', 'D', 'E']
            query = question + '\n'
            for img in images:
                if img is not None:
                    query += '<image> '
                    image_list.append(img)
            # for c, op in zip(choice_list[:len(options)], options):
            #     query += f'{c}.{op}\n'
            query += "\nDirectly give your answer, without any additional content."

            pixel_values, num_patches_list = load_images(image_list, max_num=12)
            pixel_values = pixel_values.to(torch.bfloat16).cuda()
            
            generation_config = dict(max_new_tokens=16, do_sample=False, output_attentions=True)

            start_time = time.time()
            response = model.chat(
                    tokenizer,
                    pixel_values=pixel_values,
                    num_patches_list=num_patches_list,
                    question=query,
                    generation_config=generation_config,
                )
            end_time = time.time()
            total_time += end_time - start_time

            model_answer = response.strip().lstrip('(').rstrip(')')[0].upper()
            print(model_answer)
            print(d['answer'])
            if not model_answer or model_answer not in ['A', 'B', 'C', 'D', 'E']:
                invalid += 1
            else:
                if model_answer == answer:
                    correct += 1
                else:
                    wrong += 1
        except Exception as e:
            invalid += 1
            if "out of memory" in str(e):
                oom += 1
            print(f'Error occured: {e}')
            raise(e)
            continue

    total = correct + wrong + invalid
    print(f'oom num: {oom}')
    print(f'correct answer num:{correct}\nwrong answer num:{wrong}\ninvalid answer num:{invalid}')
    print(f'correct percent:{correct / total}\nwrong percent:{wrong / total}\ninvalid percent:{invalid / total}')
    print(f'cost time:{total_time}s')
    log_file = "./log/blink.log"
    with open(log_file, 'a') as fo:
        fo.write("*"*50)
        fo.write(f"\nDesctiption:InternVL3.5-8B w/o LS")
        fo.write(f"\ntime:{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        fo.write(f'\noom num: {oom} / {total}')
        fo.write(f'\ncorrect answer num:{correct}\nwrong answer num:{wrong}\ninvalid answer num:{invalid}')
        fo.write(f'\ncorrect percent:{correct / total}\nwrong percent:{wrong / total}\ninvalid percent:{invalid / total}')
        fo.write(f'\ncost time:{total_time}s')
        fo.write('\n\n\n')

if __name__ == '__main__':
    main()