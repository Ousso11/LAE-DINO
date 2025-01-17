import os
import csv
from PIL import Image

def crop_images(img_dir, base_dir, out_dir, N=20, end='jpg'):
    # 遍历指定目录下的所有子目录
    for subdir in os.listdir(base_dir):
        # print(subdir)
        subdir_path = os.path.join(base_dir, subdir)
        print(subdir_path)
        if os.path.isdir(subdir_path):
            # crop 保存目录
            output_dir = os.path.join(out_dir, subdir_path.split('/')[-1])
            if os.path.exists(output_dir):
                print(f"已经处理过，跳过此目录文件生成{output_dir}")
                continue
            # 读取原始图像(根据实际图像格式进行修改)
            #original_image_name = subdir_path.split('/')[-1] + '.jpg'
            original_image_name = subdir_path.split('/')[-1] + '.' + end
            # original_image_name = subdir_path.split('/')[-1] + '.png'

            original_image = Image.open(os.path.join(img_dir, original_image_name)).convert('RGB')  # 假设原始图像名称为 original_image.jpg
            print(original_image_name)
            # 从CSV文件中读取裁剪信息
            targets = []
            with open(os.path.join(subdir_path, "metadata.csv"), newline='') as csvfile:  # 假设裁剪信息的文件名为 crop_info.csv
                reader = csv.DictReader(csvfile)
                for row in reader:
                    targets.append({
                        "id": int(row["id"]),
                        "area": int(row["area"]),
                        "bbox_x0": float(row["bbox_x0"]),
                        "bbox_y0": float(row["bbox_y0"]),
                        "bbox_w": float(row["bbox_w"]),
                        "bbox_h": float(row["bbox_h"]),
                        "point_input_x": float(row["point_input_x"]),
                        "point_input_y": float(row["point_input_y"]),
                        "predicted_iou": float(row["predicted_iou"]),
                        "stability_score": float(row["stability_score"]),
                        "crop_box_x0": float(row["crop_box_x0"]),
                        "crop_box_y0": float(row["crop_box_y0"]),
                        "crop_box_w": float(row["crop_box_w"]),
                        "crop_box_h": float(row["crop_box_h"]),
                    })
            ## 如果需要加速则可以注释掉下面排序的方式
            # 按照面积降序排序
            targets.sort(key=lambda x: x["area"], reverse=True)


            top_n_targets = targets[:N]

            # 遍历选择的目标信息，裁剪图像并保存
            for target in top_n_targets:
                crop_box = (
                    target["bbox_x0"],
                    target["bbox_y0"],
                    (target["bbox_x0"] + target["bbox_w"]),
                    (target["bbox_y0"] + target["bbox_h"])
                )
                x,y,x0,y0 = target["bbox_x0"], target["bbox_y0"], (target["bbox_x0"] + target["bbox_w"]), (target["bbox_y0"] + target["bbox_h"])
                center_x, center_y = (target["bbox_x0"] + target["bbox_w"] / 2), (target["bbox_y0"] + target["bbox_h"] / 2)
                cropped_image = original_image.crop(crop_box)

                # # 将图像缩放为256x256
                # cropped_image = cropped_image.resize((512, 512), Image.ANTIALIAS)

                # 确保输出目录存在
                # output_dir = os.path.join('./crop_super_output', subdir_path.split('/')[-1])
                os.makedirs(output_dir, exist_ok=True)

                crop_width, crop_height = cropped_image.size
                # print(f"crop_width: {crop_width}, crop_height: {crop_height}")
                # 保存裁剪后并缩放的图像
                if crop_width * crop_height != 0:
                    cropped_image.save(os.path.join(output_dir, f"cropped_resized_image{target['id']}_{x}_{y}_{x0}_{y0}_{center_x}_{center_y}.jpg"))
                # 显示裁剪后的图像（可选）
                # cropped_image.show()


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
    "--N",
    type=int,
    required=True,
    help="Path to either a single input image or folder of images.",
    )

    parser.add_argument(
    "--end",
    type=str,
    required=True,
    help="Path to either a single input image or folder of images.",
    )

    parser.add_argument(
    "--img_dir",
    type=str,
    required=True,
    help="Path to either a single input image or folder of images.",
    )

    parser.add_argument(
    "--base_dir",
    type=str,
    required=True,
    help="Path to either a single input image or folder of images.",
    )

    parser.add_argument(
    "--out_dir",
    type=str,
    required=True,
    help="Path to either a single input image or folder of images.",
    )

    args = parser.parse_args()

    start_time = time.time()
    
    # 指定目录
    crop_images(args.img_dir, args.base_dir, args.out_dir, args.N, args.end)

    end_time = time.time()
    runtime = end_time - start_time
    print("Function runtime:", runtime, "seconds")

