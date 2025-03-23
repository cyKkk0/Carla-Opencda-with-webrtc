#!/bin/bash

# 删除目录 inputs 和 outputs 下每个子目录中的所有文件

# 定义目录路径
INPUTS_DIR="inputs"
OUTPUTS_DIR="outputs"

# 删除 inputs 目录中的文件
if [ -d "$INPUTS_DIR" ]; then
    echo "Deleting files in $INPUTS_DIR..."
    find "$INPUTS_DIR" -type f -exec rm -f {} \;
else
    echo "$INPUTS_DIR does not exist."
fi

# 删除 outputs 目录中的文件
if [ -d "$OUTPUTS_DIR" ]; then
    echo "Deleting files in $OUTPUTS_DIR..."
    find "$OUTPUTS_DIR" -type f -exec rm -f {} \;
else
    echo "$OUTPUTS_DIR does not exist."
fi

echo "File deletion complete."
