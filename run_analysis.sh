#!/bin/bash

# --- 設定項目 ---

# モデルが保存されているディレクトリのパス
MODEL_DIR="/Users/matsuokoujirou/Documents/Data/imaging_data/Luca/Models/20251219_1622"

# 解析対象のサンプル群が入っているルートフォルダのパス
INPUT_DIR="/Users/matsuokoujirou/Documents/Data/imaging_data/Luca/Cells/Randammutant"

# WT（野生型）のデータが入っているフォルダのパス
WT_PATH="/Users/matsuokoujirou/Documents/Data/imaging_data/Luca/WT_images"

# 出力先ディレクトリ（指定しない場合は自動生成されます）
OUTPUT_DIR="/Users/matsuokoujirou/Documents/Data/imaging_data/Luca/results/plate01-02"

# --- 解析の実行 ---

# 自動モード判定: INPUT_DIR 内にサブディレクトリがあるか確認
# -mindepth 1 -maxdepth 1: 直下の要素のみ検索
# -type d: ディレクトリのみ
# -not -name '.*': 隠しディレクトリ(.DS_Store等)は除外
if [ -d "$INPUT_DIR" ] && [ -n "$(find "$INPUT_DIR" -mindepth 1 -maxdepth 1 -type d -not -name '.*')" ]; then
    echo "サブディレクトリを検出しました。FOLDER (UMAP) モードで実行します。"
    DETECTED_MODE="umap"
else
    echo "サブディレクトリが検出されませんでした。FILE モードで実行します（各ファイルを1系列として処理）。"
    DETECTED_MODE="file"
fi

# integrated_screening.py を実行します。
# 各オプションの詳細は integrated_screening.py のヘルプを参照してください。
# --umap を常時付与することで、FILEモード時でもUMAP解析を有効化します。
python integrated_screening.py \
  --mode "$DETECTED_MODE" \
  --umap \
  --model_dir "$MODEL_DIR" \
  --input_paths "$INPUT_DIR" \
  --wt_path "$WT_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --extra_viz \
  --quantitative

# 解析完了メッセージ
echo "解析が完了しました。結果は $OUTPUT_DIR を確認してください。"
