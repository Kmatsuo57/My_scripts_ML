import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from glob import glob
from tensorflow.keras.models import load_model
from csbdeep.utils import normalize
from skimage.measure import regionprops
from skimage.transform import resize
from skimage import exposure
from stardist.models import StarDist2D

# ==========================================
# 設定（先ほどと同じパスを指定してください）
# ==========================================
INPUT_DIR = "/Users/matsuokoujirou/Documents/Data/imaging_data/Luca"
# モデルが保存されているフォルダ（日付のフォルダ）を指定してください
MODEL_DIR = "/Users/matsuokoujirou/Documents/Data/imaging_data/Luca/Models/20251219_1622" 
# ↑ ここを実際の「日付フォルダ」の名前に書き換えてください！

def main():
    print(f"Loading model from: {MODEL_DIR}")
    
    # 1. 保存されたモデルの読み込み
    model_path = os.path.join(MODEL_DIR, 'best_autoencoder.keras')
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("学習が一度も成功していないか、パスが間違っています。")
        return

    try:
        autoencoder = load_model(model_path, compile=False)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. 画像データの再読み込み（可視化用に数枚あればOK）
    print("Loading a few images for visualization...")
    stardist_model = StarDist2D.from_pretrained('2D_versatile_fluo')
    file_paths = sorted(glob(os.path.join(INPUT_DIR, '*.tif')) + glob(os.path.join(INPUT_DIR, '*.tiff')))
    
    if not file_paths:
        print("Error: No images found.")
        return

    sample_cells = []
    # 最初の数ファイルだけ処理してサンプルを集める
    for file_path in file_paths[:5]: 
        try:
            image = tiff.imread(file_path)
            if image.ndim == 3 and image.shape[-1] >= 3:
                green = image[..., 1]
                seg = image[..., 2]
            else:
                green = seg = image
            
            normalized_seg = normalize(seg)
            labels, _ = stardist_model.predict_instances(normalized_seg)
            props = regionprops(labels)
            
            for prop in props:
                if prop.area < 200 or prop.area > 8000: continue
                minr, minc, maxr, maxc = prop.bbox
                cell = green[minr:maxr, minc:maxc]
                if cell.mean() < 0.5: continue
                
                cell_eq = exposure.equalize_adapthist(cell, clip_limit=0.02)
                cell_resized = resize(cell_eq, (64, 64), anti_aliasing=True)
                sample_cells.append(cell_resized)
                
                if len(sample_cells) >= 20: break # 20枚集まったら終了
        except:
            continue
        if len(sample_cells) >= 20: break

    if len(sample_cells) == 0:
        print("Error: No cells extracted.")
        return

    print(f"Extracted {len(sample_cells)} cells.")
    
    # 3. 再構成画像の生成と保存
    X = np.array(sample_cells)
    X = np.expand_dims(X, axis=-1).astype('float32')
    
    # 推論
    reconstructed = autoencoder.predict(X, verbose=0)
    
    # プロット作成
    n_samples = min(10, len(X))
    indices = np.random.choice(len(X), n_samples, replace=False)
    
    fig, axes = plt.subplots(2, n_samples, figsize=(2*n_samples, 4))
    for i, idx in enumerate(indices):
        # Original
        axes[0, i].imshow(X[idx].squeeze(), cmap='gray')
        axes[0, i].set_title('Original')
        axes[0, i].axis('off')
        
        # Reconstructed
        axes[1, i].imshow(reconstructed[idx].squeeze(), cmap='gray')
        axes[1, i].set_title('Reconstructed')
        axes[1, i].axis('off')
        
    plt.tight_layout()
    save_path = os.path.join(MODEL_DIR, 'reconstruction_samples_RETRY.png')
    plt.savefig(save_path, dpi=300)
    print(f"\nImage saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    main()