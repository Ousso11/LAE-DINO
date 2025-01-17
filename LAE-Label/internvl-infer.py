from transformers import AutoTokenizer, AutoModel
import torch
import torchvision.transforms as T
from PIL import Image
import os
import numpy as np
import re
import csv
import time
from torchvision.transforms.functional import InterpolationMode
import pandas as pd

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

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

def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=6):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def load_images(folder_path):
    image_counts = []
    image_list = []
    image_names = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            img_tensor = load_image(img_path, max_num=1).to(torch.bfloat16).cuda()
            image_count = img_tensor.size(0)
            image_counts.append(image_count)
            image_list.append(img_tensor)
            image_names.append(filename)
    pixel_values = torch.cat(image_list, dim=0)
    return pixel_values, image_counts, image_names

def get_text_by_imagename(df, imagename):
    # 查找与指定 imagename 对应的行，并返回对应的 text 值
    row = df[df['det_name'] == imagename]
    
    if not row.empty:
        return row['text'].values[0]
    else:
        return None
        
def process_folder(folder_path, csv_save_path):
    directory_name = os.path.basename(os.path.normpath(folder_path))
    csv_file = os.path.join(csv_save_path, directory_name + '.csv')
    # 检查 CSV 文件是否存在
    if os.path.exists(csv_file):
        print(f"CSV文件已存在，跳过处理：{csv_file}")
        return
    pixel_values, image_counts, image_names = load_images(folder_path)
    # print(pixel_values.shape)
    # print(image_counts)
    # import sys
    # sys.exit("程序中断执行")
    ## 如果图像大于5个
    
    generation_config = dict(
        num_beams=1,
        max_new_tokens=512,
        do_sample=False,
    )
    questions = ['Tell me the possible object category in the remote sensing image by returning a “object category” phrase surrounded by quotation marks and given a likelihood from 0 to 1 “object category” with likelihood, if it is not recognized, output "Unrecognized" and providing reasoning details.'] * len(image_counts)
    ## 正常执行直接通过internvl的输出
    responses = model.batch_chat(tokenizer, pixel_values,
                                 image_counts=image_counts,
                                 questions=questions,
                                 generation_config=generation_config)
    ## 如果需要继续提取文本，获得类别和置信度，可以注释上面的responses获得部分，直接读取csv中的text部分
    # if os.path.exists(csv_file):
    #     responses = []
    #     # 读取 CSV 文件
    #     df = pd.read_csv(csv_file)
    #     for image_name in image_names:
    #         text = get_text_by_imagename(df, image_name)
    #         responses.append(text)
    #     # print(responses)
    #     # import sys
    #     # sys.exit("程序中断执行")
        
    single_image_data = []
    for image_name, question, response in zip(image_names, questions, responses):
        class_ = re.findall(r'"([^"]*)"', response)
        match = re.search(r'with(?: a)? likelihood(?: of)? (\d+\.\d+)', response) # 有无 of 都进行检测对于InternVL2 和 1.5 版本有区别
        if match:
            likelihood = match.group(1)
        else:
            likelihood = '-1'
        try:
            one_data = (image_name, response, class_[0], likelihood)
        except Exception as e:
            return
        single_image_data.append(one_data)

    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['det_name', 'text', 'class', 'likelihood'])
        for row in single_image_data:
            writer.writerow(row)
    print(f"CSV文件写入完成：{csv_file}")


# 创建模型，然后批量处理数据
if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(
        description=(
            "Runs automatic mask generation on an input image or directory of images, "
            "and outputs masks as either PNGs or COCO-style RLEs. Requires open-cv, "
            "as well as pycocotools if saving in RLE format."
        )
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to either a single input image or folder of images.",
    )

    parser.add_argument(
        "--root_directory",
        type=str,
        required=True,
        help="Path to either a single input image or folder of images.",
    )

    parser.add_argument(
        "--csv_save_path",
        type=str,
        required=True,
        help="Path to either a single input image or folder of images.",
    )
    
    args = parser.parse_args()

    path = args.model_path
    # If you have an 80G A100 GPU, you can put the entire model on a single GPU.
    # model = AutoModel.from_pretrained(
    #     path,
    #     torch_dtype=torch.bfloat16,
    #     low_cpu_mem_usage=True,
    #     trust_remote_code=True).eval().cuda()
    # Otherwise, you need to set device_map='auto' to use multiple GPUs for inference.
    # model = AutoModel.from_pretrained(
    #     path,
    #     torch_dtype=torch.bfloat16,
    #     low_cpu_mem_usage=True,
    #     trust_remote_code=True,
    #     device_map='auto').eval()
    
    # max_memory={0: "35GiB", 1: "35GiB", 2: "35GiB", 3: "40GiB"}
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map='auto').eval()
    
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

    start = time.time()

    root_directory = args.root_directory
    csv_save_path = args.csv_save_path

    os.makedirs(csv_save_path, exist_ok=True)
    
    for folder_name in os.listdir(root_directory):
        folder_path = os.path.join(root_directory, folder_name)
        if os.path.isdir(folder_path):
            process_folder(folder_path, csv_save_path)
    end = time.time()
    print('程序运行时间为: %s Seconds' % (end - start))
