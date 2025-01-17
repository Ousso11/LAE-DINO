Step1: Remove the targets below 0.9 confidence level from the original LAE dataset and output LAE-V1 dataset.

Step2: Repeat the filtering on LAE-V1 dataset and calculate similarity by CLIP for all classes each time.

- Firstly execute 2_extract_classes.py to extract all categories of all jsonl in the folder.
- Next execute 3_merge_classes.py to merge all the categories.

Step3: Perform NMS operation on all images.