import json
import os

# 使用示例
input_directory = 'data/LAE-COD-Annotations-v4/'
output_classes_file = 'data/LAE-COD-Annotations-v4/all_classes.txt'

def extract_classes_from_jsonl(file_path):
    classes = set()

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            for instance in data['detection']['instances']:
                classes.add(instance['category'])
    
    return classes

def gather_classes_from_multiple_jsonl_files(directory):
    all_classes = set()

    # 遍历目录中的所有JSONL文件
    for filename in os.listdir(directory):
        if filename.endswith('.jsonl'):
            file_path = os.path.join(directory, filename)
            file_classes = extract_classes_from_jsonl(file_path)
            all_classes.update(file_classes)
    
    return list(all_classes)

def write_classes_to_file(classes, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as file:
        for category in classes:
            file.write(category + '\n')

all_classes = gather_classes_from_multiple_jsonl_files(input_directory)
write_classes_to_file(all_classes, output_classes_file)