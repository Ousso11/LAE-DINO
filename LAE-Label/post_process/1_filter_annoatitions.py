import json

# 使用示例
input_file_path = 'data/LAE-COD-Annotations/SLM_odvg.jsonl'
output_file_path = 'data/LAE-COD-Annotations-v1/SLM_odvg_v1.jsonl'

def read_and_filter_jsonl_file(file_path):
    results = []
    
    # 打开文件并逐行读取
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 解析JSON行
            data = json.loads(line)
            
            # 过滤目标
            filtered_instances = [
                instance for instance in data['detection']['instances']
                if float(instance['likelihood']) >= 0.9 and instance['category'] != "Unrecognized"
            ]
            
            # 如果有符合条件的实例，添加到结果中
            if filtered_instances:
                results.append({
                    'filename': data['filename'],
                    'height': data['height'],
                    'width': data['width'],
                    'detection': {
                        'instances': filtered_instances
                    }
                })
    
    return results

def write_filtered_results_to_jsonl(filtered_results, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as file:
        for result in filtered_results:
            json_line = json.dumps(result, ensure_ascii=False)
            file.write(json_line + '\n')

filtered_results = read_and_filter_jsonl_file(input_file_path)
write_filtered_results_to_jsonl(filtered_results, output_file_path)