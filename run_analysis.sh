#!/bin/bash

# --- 設定項目 ---

# モデルが保存されているディレクトリのパス
MODEL_DIR="/Users/matsuokoujirou/Documents/Data/Screening/Models/20250613_1524_improved"

# 解析対象のサンプル群が入っているルートフォルダのパス
INPUT_DIR="/Users/matsuokoujirou/Documents/Data/Screening/Abnormal_data/alldata"

# WT（野生型）のデータが入っているフォルダのパス
WT_PATH="/Users/matsuokoujirou/Documents/Data/Screening/Noemal_cells/images"

# 出力先ディレクトリ（指定しない場合は自動生成されます）
OUTPUT_DIR="/Users/matsuokoujirou/Documents/Data/Screening/screening_results/251218/Known_mutants"

# --- 解析の実行 ---

# integrated_screening.py を実行します。
# 各オプションの詳細は integrated_screening.py のヘルプを参照してください。
python integrated_screening.py \
  --mode umap \
  --model_dir "$MODEL_DIR" \
  --input_paths "$INPUT_DIR" \
  --wt_path "$WT_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --extra_viz \
  --quantitative

# 解析完了メッセージ
echo "解析が完了しました。結果は $OUTPUT_DIR を確認してください。"
