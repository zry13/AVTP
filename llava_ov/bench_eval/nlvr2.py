import argparse
import torch

from llava_avtp.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava_avtp.conversation import conv_templates, SeparatorStyle
from llava_avtp.model.builder import load_pretrained_model
from llava_avtp.utils import disable_torch_init
from llava_avtp.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re
from datasets import load_dataset
from tqdm import tqdm
import time

def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if isinstance(image_file, Image.Image):
        image = image_file.convert("RGB")
    elif image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

def extract_yes_or_no(model_answer):
    model_answer = model_answer.lower()
    yes_pos = model_answer.find('yes')
    no_pos = model_answer.find('no')
    
    if yes_pos == -1 and no_pos == -1:
        return None
    
    if  yes_pos != -1 and no_pos == -1:
        return 'yes'
    
    if yes_pos == -1 and no_pos != -1:
        return 'no'
    
    return 'yes' if yes_pos < no_pos else 'no'

def main():
    # Model
    model_path = "llava-onevision-qwen2-7b-ov-chat"
    vit_path = "siglip-so400m-patch14-384"
    model_name = "llava_qwen"
    device = "cuda"

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=model_name,
        mm_vision_tower=vit_path,
        mm_vision_select_feature="patch",
        attn_implementation="flash_attention_2",
        device_map="auto"
    )
    model.eval()

    correct, wrong, invalid = 0, 0, 0
    total_time = 0
    oom = 0

    ds = load_dataset("lmms-lab/NLVR2")

    for i, d in enumerate(tqdm(ds['test'])):
        # if i >= 5:
        #     break
        try:
            q = d['question']
            images = d['images']
            options = d['options']
            answer = d['answer']

            query = q + '\n' + str(options)


            query += f'\nAnswer the question directly, without any additional content.'

            images = load_images(images)
            image_sizes = [x.size for x in images]
            image_tensor = process_images(
                images,
                image_processor,
                model.config
            )
            images_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

            # conv = conv_templates["llava_v1"].copy()
            conv = conv_templates["qwen_1_5"].copy()
            # conv.system = ("A chat between a curious human and an artificial intelligence assistant. "
            #         "The assistant gives helpful, detailed, and polite answers to the human's questions. "
            # )
            conv.append_message(conv.roles[0], query)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = (
                tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                .unsqueeze(0)
                .cuda()
            )

            with torch.inference_mode():
                start_time = time.time()
                output_ids = model.generate(
                    inputs=input_ids,
                    images=images_tensor,
                    image_sizes=image_sizes,
                    do_sample=False,
                    # temperature=0.2,
                    max_new_tokens=16,
                    use_cache=True,
                    pad_token_id = tokenizer.eos_token_id,
                    # return_dict_in_generate=True,
                    # output_attentions=True,
                    # output_hidden_states=True
                )
                end_time = time.time()
                total_time += end_time - start_time
            
            # sequences = output_ids['sequences'][:, input_ids.size(1):]
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            print(f"model answer:{outputs}")
            print(f"gt answer:{d['answer']}")
            model_answer = outputs.strip().lstrip('(').rstrip(')')[0].upper()
            if not model_answer or model_answer not in ['A', 'B', 'C', 'D']:
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
            # raise(e)
            continue

    total = correct + wrong + invalid
    print(f'oom num: {oom}')
    print(f'correct answer num:{correct}\nwrong answer num:{wrong}\ninvalid answer num:{invalid}')
    print(f'correct percent:{correct / total}\nwrong percent:{wrong / total}\ninvalid percent:{invalid / total}')
    print(f'cost time:{total_time}s')
    log_file = "./llava_ov/log/nlvr2.log"
    with open(log_file, 'w') as fo:
        fo.write(f'oom num: {oom} / {total}')
        fo.write(f'\ncorrect answer num:{correct}\nwrong answer num:{wrong}\ninvalid answer num:{invalid}')
        fo.write(f'\ncorrect percent:{correct / total}\nwrong percent:{wrong / total}\ninvalid percent:{invalid / total}')
        fo.write(f'\ncost time:{total_time}s')


if __name__ == "__main__":
    main()
