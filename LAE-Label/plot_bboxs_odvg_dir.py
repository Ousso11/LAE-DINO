import os
import random
import cv2
import json

def plot_one_box(x, img, color=None, label=None, line_thickness=2):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        # c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        # cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        # cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

        c2 = c1[0] + t_size[0], c1[1] + t_size[1] + 5
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] + 15), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def draw_boxes(image_path, annotations_path):
    image = cv2.imread(image_path)

    with open(annotations_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            # Check if the current file is the one we're interested in
            if data['filename'] == os.path.basename(image_path):
                for instance in data['detection']['instances']:
                    bbox = instance['bbox']
                    label = instance['category']
                    plot_one_box(bbox, image, label=label)

    # Save the annotated image
    save_path = os.path.join(os.path.dirname(image_path), 'annotated_images')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    output_file_name = os.path.splitext(os.path.basename(image_path))[0] + '_annotated.jpg'
    output_file_path = os.path.join(save_path, output_file_name)

    cv2.imwrite(output_file_path, image)
    print(f"Annotated image saved to {output_file_path}")

# Function to get all image files in a directory
def get_image_files(directory):
    image_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                image_files.append(os.path.join(root, file))
    return image_files

# Main function to process all images in a directory
def process_images_in_directory(directory, annotations_path):
    image_files = get_image_files(directory)
    for image_path in image_files:
        draw_boxes(image_path, annotations_path)

# Usage example
image_directory = './data/DATASET/images/'  # Replace with the actual directory path
annotations_path = './data/DATASET/DATASET.jsonl'  # Replace with the actual JSONL file path
process_images_in_directory(image_directory, annotations_path)