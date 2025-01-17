import argparse
import os
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import base64
import csv
import re
from openai import OpenAI

def encode_image_base64(image: Image.Image) -> str:
    """Encode image to base64 format."""
    buffered = BytesIO()
    image.save(buffered, format='JPEG')
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def get_subdirectories(input_dir):
    """Get all subdirectories in the input directory."""
    subdirectories = [os.path.join(input_dir, sub_dir) for sub_dir in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, sub_dir))]
    return subdirectories

def get_file_list(input_dir):
    """Get a list of image files in the directory."""
    file_list = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                file_list.append(os.path.join(root, file))
    return file_list

def parse_args():
    parser = argparse.ArgumentParser(description="OpenAI Infer")
    parser.add_argument("--api_key", type=str, default="", help="OpenAI API Key")
    parser.add_argument("--base_url", type=str, default="", help="OpenAI Base URL")
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo", help="OpenAI Model")
    parser.add_argument("--input_dir", type=str, default="", help="Input Directory")
    parser.add_argument("--output_dir", type=str, default="", help="Output Directory")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature")
    parser.add_argument("--top_p", type=float, default=0.7, help="Top_p")
    args = parser.parse_args()
    return args

def process_images_in_subdirectory(subdirectory, client, model_name, prompt, temperature, top_p, output_dir):
    """Process images in a given subdirectory and write results to a CSV file."""
    file_list = get_file_list(subdirectory)
    result_list = []
    for file in tqdm(file_list, desc=f"Processing {subdirectory}"):
        base64_image = encode_image_base64(Image.open(file))
        messages = [{
            'role': 'user',
            'content': [{'type': 'text', 'text': prompt},
                        {'type': 'image_url',
                         'image_url': {
                             "url": f"data:image/jpeg;base64,{base64_image}",
                             "detail": "low"
                         },
                         }],
        }]
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
        )
        response = response.choices[0].message.content
        class_ = re.findall(r'"([^"]*)"', response)
        match = re.search(r'with a likelihood of (\d+\.\d+)', response)
        likelihood = match.group(1) if match else '-1'
        file_name = os.path.basename(file)
        
        try:
            one_data = (file_name, response, class_[0], likelihood)
        except Exception as e:
            one_data = ('-1', '-1', '-1', '-1')
            print(f"Error processing {file}: {e}")
        result_list.append(one_data)

    directory_name = os.path.basename(os.path.normpath(subdirectory))
    csv_file = os.path.join(output_dir, directory_name + '.csv')
    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['det_name', 'text', 'class', 'likelihood'])
        for row in result_list:
            writer.writerow(row)
    print(f"CSV file written: {csv_file}")

def main(args):
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    model_name = args.model_name
    prompt = 'Tell me the possible object category in the remote sensing image by returning a “object category” phrase surrounded by quotation marks and given a likelihood from 0 to 1 “object category” with likelihood, if it is not recognized, output "Unrecognized" and providing reasoning details.'
    os.makedirs(args.output_dir, exist_ok=True)
    subdirectories = get_subdirectories(args.input_dir)
    for subdirectory in subdirectories:
        directory_name = os.path.basename(os.path.normpath(subdirectory))
        csv_file = os.path.join(args.output_dir, directory_name + '.csv')
        if os.path.exists(csv_file):
            print(f"{csv_file} exists")
            continue
        else:
            process_images_in_subdirectory(subdirectory, client, model_name, prompt, args.temperature, args.top_p, args.output_dir)

if __name__ == "__main__":
    args = parse_args()
    main(args)
