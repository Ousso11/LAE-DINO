import csv
import jsonlines
import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

lock = Lock()  # Lock to ensure thread-safe operations on the shared data list

def get_image_dimensions(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
        return width, height

# Read the CSV file and convert to JSON Lines format
def read_csv_to_json(img_dir, csv_file, end, writer):
    filename_new = os.path.basename(os.path.normpath(csv_file)).split('.')[0]
    # file_path = os.path.join(img_dir, filename_new + '.png')
    # file_path = os.path.join(img_dir, filename_new + '.jpg')
    # file_path = os.path.join(img_dir, filename_new + '.tif')
    file_path = os.path.join(img_dir, filename_new + '.' + end)
    print(file_path)
    width, height = get_image_dimensions(file_path)
    image_info = {
        # "filename": filename_new + '.png',
        # "filename": filename_new + '.jpg',
        # "filename": filename_new + '.tif',
        "filename": filename_new + '.' + end,
        "height": height,
        "width": width,
        "detection": {
            "instances": []
        }
    }
    data = []
    with open(csv_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        instances = []
        for row in reader:
            image_filename = row['det_name'].split('.jpg')[0]
            print(image_filename)
            try:
                x, y, x0, y0, x_center, y_center = (float(image_filename.split('_')[-6]),
                                                    float(image_filename.split('_')[-5]),
                                                    float(image_filename.split('_')[-4]),
                                                    float(image_filename.split('_')[-3]),
                                                    float(image_filename.split('_')[-2]),
                                                    float(image_filename.split('_')[-1]))
            
                bbox = [x, y, x0, y0]
                instance = {
                    "bbox": bbox,
                    "category": row['class'],
                    "likelihood": row['likelihood']
                }
                instances.append(instance)
            except Exception as e:
                print(f"Error processing: {e}")
        image_info["detection"] = {"instances": instances}
        data.append(image_info)
        # print(image_info["filename"])

    with lock:
        writer.write_all(data)  # Safely write data to the shared JSON Lines file

def process_csv_files(img_dir, base_csv_dir, output_file):
    csv_files = [os.path.join(base_csv_dir, file) for file in os.listdir(base_csv_dir) if file.endswith('.csv')]
    # 获取原始图像的后缀名称
    file_exp = os.listdir(img_dir)[0]
    end = file_exp.split('.')[-1]
    
    with jsonlines.open(output_file, mode="w") as writer:
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for csv_file in csv_files:
                futures.append(executor.submit(read_csv_to_json, img_dir, csv_file, end, writer))
            
            for future in as_completed(futures):
                future.result()  # Handle any exceptions raised during execution

img_directory = './data/LAE-COD/DATASET/img/'
csv_base_directory = './data/LAE-COD/DATASET/csv/'
output_file = './data/LAE-COD/DATASET/DATASET_ovdg.jsonl'

# Process CSV files and write data incrementally to the JSON Lines file
process_csv_files(img_directory, csv_base_directory, output_file)

print(f"All data written to {output_file}.")
