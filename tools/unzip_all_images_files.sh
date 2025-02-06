#!/bin/bash

# 将 LAE-Dataset 重命名为 data
if [ -d "LAE-Dataset" ]; then
    mv LAE-Dataset data
else
    echo "Error: Directory 'LAE-Dataset' not found."
    exit 1
fi

# 遍历 data 目录下的 LAE-COD 和 LAE-FOD 目录
for subdir in "data/LAE-COD" "data/LAE-FOD"; do
    if [ -d "$subdir" ]; then
        echo "Processing directory: $subdir"
        
        # 遍历并解压 zip 文件
        for zipfile in "$subdir"/*.zip; do
            if [ -f "$zipfile" ]; then
                echo "Unzipping $zipfile"
                unzip -o "$zipfile" -d "$subdir"
            fi
        done
    else
        echo "Warning: Directory '$subdir' not found."
    fi
done

echo "All zip files have been processed."