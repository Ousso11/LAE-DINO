from PIL import Image
import os
import concurrent.futures

def crop_image_to_tiles(file_path, output_path, tile_size=(1024, 1024)):
    """
    Crop a single image into tiles and save the tiles to the output_path.
    """
    file_name = os.path.basename(file_path)
    with Image.open(file_path) as img:
        img_width, img_height = img.size

        # Calculate the number of tiles in both dimensions
        x_tiles = img_width // tile_size[0]
        y_tiles = img_height // tile_size[1]

        # Generate cropped tiles
        for x in range(x_tiles):
            for y in range(y_tiles):
                # Define the coordinates of the rectangle to crop
                left = x * tile_size[0]
                upper = y * tile_size[1]
                right = left + tile_size[0]
                lower = upper + tile_size[1]

                # Crop the image
                if right <= img_width and lower <= img_height:
                    cropped_img = img.crop((left, upper, right, lower))
                    # Save the cropped image
                    cropped_img.save(os.path.join(output_path, f"{file_name[:-4]}_{x}_{y}.png"))

def process_images(input_path, output_path, tile_size=(1024, 1024)):
    """
    Process all images in the input_path using multithreading.
    """
    # Check if output path exists, if not, create it
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # List all image files in the input directory
    image_files = [
        os.path.join(input_path, file_name) for file_name in os.listdir(input_path)
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
    ]

    # Use ThreadPoolExecutor to process images in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(crop_image_to_tiles, file_path, output_path, tile_size) for file_path in image_files]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing image: {e}")

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
        "--input_folder",
        type=str,
        required=True,
        help="Path to either a single input image or folder of images.",
    )

    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Path to either a single input image or folder of images.",
    )

    args = parser.parse_args()

    start_time = time.time()
    
    # Crop images using multithreading
    process_images(args.input_folder, args.output_folder)

    end_time = time.time()
    runtime = end_time - start_time
    print("Function runtime:", runtime, "seconds")
