import os
import json
import numpy as np
import cv2
folder_path = 'data/LAE-COD-Annotations-v4'

def read_jsonl_files(folder_path):
    jsonl_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.jsonl')]
    all_files_data = []
    for file_path in jsonl_files:
        file_data = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                file_data.append(json.loads(line.strip()))
        all_files_data.append(file_data)
    return all_files_data

def apply_nms_for_category(data, iou_threshold=0.5):
    boxes = np.array([item['bbox'] for item in data])
    scores = np.array([item['likelihood'] for item in data],dtype=np.float32)
    
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), score_threshold=0.5, nms_threshold=iou_threshold)
    # indices is a list of indices that survived NMS
    return [data[i] for i in indices.flatten()]

def nms_on_same_category_data(data, iou_threshold=0.5):
    category_dict = {}
    for item in data['detection']['instances']:
        category = item['category']
        if category not in category_dict:
            category_dict[category] = []
        category_dict[category].append(item)
    
    result = []
    for category, items in category_dict.items():
        result.extend(apply_nms_for_category(items, iou_threshold))
    
    return result

if __name__ == "__main__":
    all_data = read_jsonl_files(folder_path)
    all_file_names = [f for f in os.listdir(folder_path) if f.endswith('.jsonl')]
    for data, file_name in zip(all_data, all_file_names):
        nms_results = nms_on_same_category_data(data[0])