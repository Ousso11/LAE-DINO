import os
import clip
import torch
import numpy as np
import json
input_txt_path = "data/LAE-COD-Annotations-v3/all_classes.txt"

input_jsonl_folder = "data/LAE-COD-Annotations-v3"
output_jsonl_folder = "data/LAE-COD-Annotations-v4"

output_jsonl = "data/class_map_v3.jsonl"
simularity_thr = 0.92


def read_lines_from_txt(file_path):
    lines = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            lines.append(line.strip())
    return lines


def read_class_mappings(file_path):
    mappings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            mappings[data["source"]] = data["target"]
    return mappings


def update_category_mappings(input_folder, output_folder, mappings):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        if file_name.endswith(".jsonl"):
            input_file_path = os.path.join(input_folder, file_name)
            output_file_path = os.path.join(output_folder, file_name)

            with open(input_file_path, 'r', encoding='utf-8') as f_in, open(output_file_path, 'w', encoding='utf-8') as f_out:
                for line in f_in:
                    data = json.loads(line)
                    for instance in data["detection"]["instances"]:
                        category = instance["category"]
                        # 如果在映射中，更新类别
                        if category in mappings:
                            instance["category"] = mappings[category]
                    # 保存更新后的json数据
                    json.dump(data, f_out, ensure_ascii=False)
                    f_out.write('\n')


if __name__ == "__main__":
    classes = read_lines_from_txt(input_txt_path)
    # load clip model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, prepross = clip.load("ViT-B/32", device=device)
    # tokenize classes
    classes_tokenized = clip.tokenize(classes).to(device)
    # calculate simularity map
    with torch.no_grad():
        classes_features = model.encode_text(classes_tokenized)
        simularity_map = torch.nn.functional.cosine_similarity(
            classes_features.unsqueeze(1), classes_features.unsqueeze(0), dim=-1
        ) - torch.eye(classes_features.size(0)).to(device)
        max_values, max_indices = torch.max(simularity_map, dim=-1)
        source_class = classes
        target_class = [classes[i] for i in max_indices.cpu().numpy()]
        results = []
        processed_classes = []
        for idx, (source_class, target_class, simularity) in enumerate(zip(source_class,
                                                                           target_class,
                                                                           max_values,
                                                                           )):
            if simularity > simularity_thr:
                if source_class in processed_classes:
                    continue
                else:
                    results.append(
                        {
                            "source": source_class,
                            "target": target_class,
                            "simularity": float(simularity)
                        }
                    )
                    processed_classes.append(source_class)
                    processed_classes.append(target_class)
    with open(output_jsonl, 'w', encoding='utf-8') as f_jsonl:
        for result in results:
            json.dump(result, f_jsonl, ensure_ascii=False)
            f_jsonl.write('\n')
    # 新增：读取映射结果并更新类别
    mappings = read_class_mappings(output_jsonl)
    update_category_mappings(input_jsonl_folder, output_jsonl_folder, mappings)
