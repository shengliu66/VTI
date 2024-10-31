import argparse
import torch
import os
import json
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from PIL import Image
from transformers import set_seed

from vti_utils.utils import get_demos, obtain_textual_vti, obtain_visual_vti
from vti_utils.llm_layers import add_vti_layers, remove_vti_layers

from datasets import load_dataset

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def eval_model(args):
    # Model
    # disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    input_images, input_ids = get_demos(args, image_processor, model, tokenizer)

    torch.cuda.empty_cache()

    print('Obtaining direction\n')

    qs = ''
    cur_prompt = qs
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    vision_token_start = len(tokenizer(prompt.split("<image>")[0], return_tensors='pt')["input_ids"][0])
    vision_token_end = vision_token_start + model.get_vision_tower().num_patches

    if args.alpha_image != 0:
        vti_vision, _ = obtain_visual_vti(
            model, input_images, rank=1
            )

        visual_direction = vti_vision[1:]

    if args.alpha_image != 0:
        add_vti_layers(model.model.vision_tower.vision_tower.vision_model, torch.stack([visual_direction],dim=1).cuda(), alpha = [args.alpha_image])
    
    if args.alpha_text != 0:

        vti_text, _ = obtain_textual_vti(
            model, input_ids, input_images, rank=1
            )
        textual_direction = vti_text[1:]

    if args.alpha_text != 0:
        add_vti_layers(model, torch.stack([textual_direction],dim=1).cuda(), alpha = [args.alpha_text])

    torch.cuda.empty_cache()

    
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)


    dataset = load_dataset("Shengcao1006/MMHal-Bench")['test']


    ans_file = open(answers_file, "w")
    for img_id in range(len(dataset)):
        image_path = dataset[img_id]['image_path']
        raw_image = load_image(image_path)
        qs = dataset[img_id]['question']
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image_tensor = image_processor.preprocess(raw_image, return_tensors='pt')['pixel_values'][0]

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        img_save = {}

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                num_beams=5,
                max_new_tokens=256,
                do_sample=args.sample,
                use_cache=False)

        outputs = tokenizer.batch_decode(output_ids[:, :], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()


        img_save["question_type"] = dataset[img_id]["question_type"]
        img_save["question_topic"] = dataset[img_id]["question_topic"]
        img_save["image_id"] = dataset[img_id]["image_id"]
        img_save["image_src"] = dataset[img_id]["image_src"]
        img_save["image_content"] = dataset[img_id]["image_content"]
        img_save["question"] = dataset[img_id]["question"]
        img_save["gt_answer"] = dataset[img_id]["gt_answer"]
        img_save["model_answer"] = outputs

        ans_file.write(json.dumps(img_save) + "\n")
        ans_file.flush()
    ans_file.close()
    remove_vti_layers(model)
    remove_vti_layers(model.model.vision_tower.vision_tower.vision_model)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/data/datasets/MSCOCO/val2014")
    parser.add_argument("--answers-file", type=str, default="/results/coco_pope_popular_answer.jsonl")
    parser.add_argument("--data-file", type=str, default="/data/datasets/MSCOCO/")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num_demos", type=int, default=50)
    parser.add_argument("--alpha_image", type=float, default=0)
    parser.add_argument("--alpha_text", type=float, default=0.8)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--sample", action='store_true')

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mask_ratio", type=float, default=0.99)
    parser.add_argument("--num_trials", type=int, default=50)
    
    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)