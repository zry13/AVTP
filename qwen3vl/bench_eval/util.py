import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# from modelscope import snapshot_download
# from qwen_vl_utils import process_vision_info
# from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

def warm_up(model, processor):
    if model.config.model_type == "qwen3_vl":
        for i in range(10):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                        },
                        {"type": "text", "text": "Describe this image."},
                    ],
                }
            ]
            # Preparation for inference
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            inputs = inputs.to(model.device)

            # Inference: Generation of the output
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            print(f"Warm up step:{i}:{output_text[0]}")
        print(f"Warm up done.")
    else:
        raise("Unsupported model.")

def main():
    from datasets import load_dataset
    from tqdm import tqdm
    ds = load_dataset("MUIRBENCH/MUIRBENCH", "default")
    img_count = {}
    for i, d in enumerate(tqdm(ds['test'])):
        image_list = d['image_list']
        img_cnt = len(image_list)
        if img_cnt not in img_count.keys():
            img_count[img_cnt] = 1
        else:
            img_count[img_cnt] += 1
    print(img_count)

if __name__ == '__main__':
    main()