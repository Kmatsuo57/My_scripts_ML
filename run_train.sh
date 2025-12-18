#!/bin/bash

# ==========================================
# Training Configuration
# ==========================================

# 入力画像のフォルダパス（最後にスラッシュをつけない）
INPUT_DIR="/Users/matsuokoujirou/Documents/Data/Screening/Noemal_cells/images"

# 出力先フォルダのルートパス
# （実行時に日付入りサブフォルダが作成されるようにPython側で処理しても良いが、
#   ここでは分かりやすく引数として日付フォルダを指定する）
DATE_STR=$(date "+%Y%m%d_%H%M")
OUTPUT_DIR="/Users/matsuokoujirou/Documents/Data/Screening/Models/${DATE_STR}"

# ==========================================
# Execution
# ==========================================

echo "Starting Autoencoder Training..."
echo "Input:  ${INPUT_DIR}"
echo "Output: ${OUTPUT_DIR}"

# Pythonスクリプトの実行
python CAE_improved_modeltrain.py \
    --input_dir "${INPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}"

# エラーハンドリング
if [ $? -eq 0 ]; then
    echo "Training finished successfully!"
else
    echo "Training failed!"
    exit 1
fi