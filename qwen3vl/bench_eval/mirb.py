import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from modelscope import snapshot_download
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from ..modeling_qwen3vl_avtp import Qwen3VLForConditionalGeneration
import re
from datasets import load_dataset
from tqdm import tqdm
import time
from datetime import datetime
from gpt4o import gpt4o_judge
import json
from util import warm_up

def get_messages(query, images):
    content = []
    image_index = 0
    parts = query.split("<image>")
    for i, part in enumerate(parts):
        if i == 0 and not part:
            
            if image_index < len(images):
                content.append({
                    "type": "image",
                    "image": images[image_index]
                })
                image_index += 1
            continue
            
        if part:
            content.append({
                "type": "text",
                "text": part
            })
        
        if i < len(parts) - 1:
            if image_index < len(images):
                content.append({
                    "type": "image",
                    "image": images[image_index]
                })
                image_index += 1
    
    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]

    return messages

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
    model_path = "Qwen3-VL-8B-Instruct"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, dtype="auto", device_map="auto",
        attn_implementation='flash_attention_2',
        # attn_implementation='eager',
    )
    processor = AutoProcessor.from_pretrained(model_path)
    model.eval()

    correct, wrong, invalid = 0, 0, 0
    total_time = 0
    oom = 0

    warm_up(model, processor)

    data_path = "./MIRB"
    data_file = ["3d_scene.json", "analogy.json", "arxiv.json", "attribute.json", "codeu.json", "count.json", "food.json", \
        "image_jigsaw.json", "plot_code.json", "plot_text.json", "sightseeing.json", "visual_chain.json"]
    ds = []
    for df in data_file:
        file_path = os.path.join(data_path, df)
        with open(file_path, 'r', encoding='utf-8') as fi:
            data = json.load(fi)
        ds += data

    for i, d in enumerate(tqdm(ds[:])):
        try:
            question = d['questions']
            answer = d['answers']
            
            images = [os.path.join(data_path, img_path) for img_path in d['images']]
            
            for img in images:
                question += "\n<image>"

            query = question + "\nAnswer the question directly, without any additional content."

            messages = get_messages(query, images)

            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            inputs = inputs.to(model.device)

            start_time = time.time()
            generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False, temperature=0.0)
            end_time = time.time()
            total_time += end_time - start_time

            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            print(f"model answer:{output_text}")
            print(f"gt answer:{d['answers']}")
            model_answer = output_text.strip().lstrip('(').rstrip(')')
            model_answer = str(model_answer)
            answer = str(answer)
            if len(answer) == 1 and str(answer).isalpha():
                model_answer = model_answer[0].upper()
                if not model_answer or model_answer not in ['A', 'B', 'C', 'D', 'E']:
                    invalid += 1
                else:
                    if model_answer == answer:
                        correct += 1
                    else:
                        wrong += 1
            elif answer.lower() == 'yes' or answer.lower() == 'no':
                model_answer = extract_yes_or_no(model_answer.lower())
                if model_answer is None:
                    invalid += 1
                if model_answer == answer:
                    correct += 1
                else:
                    wrong += 1
            else:
                if model_answer is None:
                    invalid += 1
                else:
                    judge = gpt4o_judge(model_answer, answer)
                    if not judge:
                        invalid += 1
                    else:
                        if judge == 'yes':
                            correct += 1
                        else:
                            wrong += 1
            
        except Exception as e:
            invalid += 1
            if "out of memory" in str(e):
                oom += 1
                print(f'Error occured: {e}')
                continue
            raise(e)
            continue
    
    total = correct + wrong + invalid
    print(f'oom num: {oom}')
    print(f'correct answer num:{correct}\nwrong answer num:{wrong}\ninvalid answer num:{invalid}')
    print(f'correct percent:{correct / total}\nwrong percent:{wrong / total}\ninvalid percent:{invalid / total}')
    print(f'cost time:{total_time}s')
    log_file = "./log/mirb.log"
    with open(log_file, 'a') as fo:
        fo.write("*"*50)
        fo.write(f"\nDesctiption:qwen3-vl-8b v2drop keep ratio=0.05")
        fo.write(f"\ntime:{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        fo.write(f'\noom num: {oom} / {total}')
        fo.write(f'\ncorrect answer num:{correct}\nwrong answer num:{wrong}\ninvalid answer num:{invalid}')
        fo.write(f'\ncorrect percent:{correct / total}\nwrong percent:{wrong / total}\ninvalid percent:{invalid / total}')
        fo.write(f'\ncost time:{total_time}s')
        fo.write('\n\n\n')

if __name__ == '__main__':
    main()