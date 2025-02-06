#!/bin/bash

# 遍历 data 目录下的 LAE-COD 和 LAE-FOD 目录
for maindir in "data/LAE-COD" "data/LAE-FOD"; do
    if [ -d "$maindir" ]; then
        echo "Processing main directory: $maindir"
        
        # 遍历主目录下的所有子目录
        for subdir in "$maindir"/*/ ; do
            if [ -d "$subdir" ]; then
                echo "Processing subdirectory: $subdir"
                
                # 遍历并解压子目录中的 zip 文件
                for zipfile in "$subdir"*.zip; do
                    if [ -f "$zipfile" ]; then
                        echo "Unzipping $zipfile"
                        unzip -o "$zipfile" -d "$subdir"
                    fi
                done
            fi
        done
    else
        echo "Warning: Directory '$maindir' not found."
    fi
done

echo "All zip files have been processed."
