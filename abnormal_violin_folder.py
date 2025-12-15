import numpy as np
import tifffile as tiff
import os
from glob import glob
from tensorflow.keras.models import load_model
from stardist.models import StarDist2D
from csbdeep.utils import normalize
from skimage.measure import regionprops
from skimage.transform import resize
from skimage import exposure
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime
import pickle

class ProductionMutantScreeningFolder:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.load_trained_models()
        
    def load_trained_models(self):
        print("Loading trained models...")
        self.autoencoder = load_model(os.path.join(self.model_dir, 'best_autoencoder.keras'))
        self.encoder = load_model(os.path.join(self.model_dir, 'encoder.keras'))
        
        with open(os.path.join(self.model_dir, 'scaler.pkl'), 'rb') as f:
            self.scaler = pickle.load(f)
        with open(os.path.join(self.model_dir, 'pca.pkl'), 'rb') as f:
            self.pca = pickle.load(f)
        
        with open(os.path.join(self.model_dir, 'detector_conservative.pkl'), 'rb') as f:
            self.detector_conservative = pickle.load(f)
        with open(os.path.join(self.model_dir, 'detector_moderate.pkl'), 'rb') as f:
            self.detector_moderate = pickle.load(f)
        
        try:
            self.stardist_model = StarDist2D.from_pretrained('2D_versatile_fluo')
        except:
            print("Downloading StarDist model...")
            self.stardist_model = StarDist2D.from_pretrained('2D_versatile_fluo')
        print("Models loaded.")

    def get_subfolders(self, root_folder):
        """ルートフォルダ直下のサブフォルダを辞書として取得"""
        if not os.path.exists(root_folder):
            print(f"Root folder not found: {root_folder}")
            return {}
        
        subfolders = [f.path for f in os.scandir(root_folder) if f.is_dir()]
        folders_dict = {os.path.basename(f): f for f in subfolders}
        
        # もしサブフォルダがなく、直下にTIFがある場合は、そのフォルダ自身を1つのサンプルとして扱うか、
        # あるいは「ルート指定ミス」として警告する。今回はサブフォルダ探索モード。
        print(f"Found {len(folders_dict)} subfolders in {root_folder}")
        return folders_dict

    def extract_quality_cells(self, image_path, enhance_contrast=True):
        """
        細胞抽出
        enhance_contrast=True: 解析用
        enhance_contrast=False: 表示用(Raw)
        """
        try:
            image = tiff.imread(image_path)
            if image.ndim == 3 and image.shape[-1] >= 3:
                seg_c, green_c = image[..., 2], image[..., 1]
            else:
                seg_c, green_c = image, image
            
            normalized_seg = normalize(seg_c)
            labels, _ = self.stardist_model.predict_instances(normalized_seg)
            props = regionprops(labels)
            
            cells = []
            h, w = labels.shape
            img_max = np.max(green_c) if np.max(green_c) > 0 else 1
            
            for prop in props:
                minr, minc, maxr, maxc = prop.bbox
                if (minr < 10 or minc < 10 or maxr > (h - 10) or maxc > (w - 10)): continue
                if prop.area < 200 or prop.area > 8000: continue
                if prop.eccentricity > 0.95: continue
                
                cell_img = green_c[minr:maxr, minc:maxc]
                if np.mean(cell_img) < 0.5 or np.std(cell_img) < 0.1: continue
                
                if enhance_contrast:
                    # 解析用: コントラスト強調
                    cell_float = cell_img / np.max(cell_img) if np.max(cell_img) > 0 else cell_img
                    cell_eq = exposure.equalize_adapthist(cell_float, clip_limit=0.02)
                    cell_final = resize(cell_eq, (64, 64), anti_aliasing=True)
                else:
                    # 表示用: Raw (リサイズのみ)
                    cell_resized = resize(cell_img, (64, 64), anti_aliasing=True, preserve_range=True)
                    cell_final = cell_resized / img_max
                
                cells.append(cell_final)
            return cells
        except Exception as e:
            print(f"Error extracting from {os.path.basename(image_path)}: {e}")
            return []

    def compute_scores(self, cell_images):
        if not cell_images: return {}
        X = np.expand_dims(np.array(cell_images), axis=-1).astype('float32')
        
        # MSE
        reconstructed = self.autoencoder.predict(X, verbose=0)
        mse = np.mean(np.square(X - reconstructed), axis=(1, 2, 3))
        
        # Anomaly Detection
        encoded = self.encoder.predict(X, verbose=0).reshape(len(X), -1)
        encoded_pca = self.pca.transform(self.scaler.transform(encoded))
        
        cons_pred = self.detector_conservative.predict(encoded_pca)
        mod_pred = self.detector_moderate.predict(encoded_pca)
        cons_score = -self.detector_conservative.decision_function(encoded_pca)
        
        return {
            'mse': mse,
            'cons_pred': cons_pred,
            'mod_pred': mod_pred,
            'cons_score': cons_score,
            'cons_rate': np.sum(cons_pred == -1) / len(cons_pred)
        }

    def screen_folders(self, folders_dict, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print("\n=== Starting Folder-Based Screening ===")
        
        results = {}
        detailed_results = []
        
        # 1. WT Baseline (WTフォルダを探す)
        wt_path = None
        for name, path in folders_dict.items():
            if name.upper() == 'WT': wt_path = path; break
            
        if not wt_path:
            # 固定パスフォールバック（必要に応じて変更）
            fixed_wt = "/Users/matsuokoujirou/Documents/Data/Screening/Pools/normal/WT" # フォルダパスであること
            # 固定パスが画像ファイルの場合は、ファイル単体処理用のロジックが必要だが、
            # ここでは「フォルダベース」なのでWTもフォルダであることを期待するか、
            # あるいは「WT画像ファイル」を一時的に読み込む特例処理が必要。
            # 簡易化のため「WTフォルダ」が存在することを前提とします。
            if os.path.exists(fixed_wt) and os.path.isdir(fixed_wt):
                wt_path = fixed_wt
                print(f"Using fixed WT folder: {wt_path}")
            else:
                print("Warning: WT folder not found. Using default thresholds.")
        
        wt_thresholds = {'rate': 0.0, 'threshold': 5.0}
        
        if wt_path:
            print(f"Calculating Baseline from: {os.path.basename(wt_path)}")
            # WTフォルダ内の全画像を処理
            wt_tif_files = sorted(glob(os.path.join(wt_path, '*.tif')))
            wt_cells_all = []
            for f in wt_tif_files:
                wt_cells_all.extend(self.extract_quality_cells(f, enhance_contrast=True))
            
            if wt_cells_all:
                wt_scores = self.compute_scores(wt_cells_all)
                rate = wt_scores['cons_rate'] * 100
                wt_thresholds = {'rate': rate, 'threshold': rate + 4.2}
                print(f"  WT Baseline: {rate:.2f}% | Threshold: {wt_thresholds['threshold']:.2f}%")

        # 2. 各フォルダの処理
        total_folders = len(folders_dict)
        for idx, (name, folder_path) in enumerate(folders_dict.items(), 1):
            print(f"[{idx}/{total_folders}] Folder: {name}...", end='\r')
            
            tif_files = sorted(glob(os.path.join(folder_path, '*.tif')))
            if not tif_files: continue
            
            # フォルダ内の全細胞を集める
            folder_cells_enhanced = []
            cell_metadata = [] # (file_path, local_index) を記録して、後でRaw画像を取り出せるようにする
            
            for f_path in tif_files:
                cells = self.extract_quality_cells(f_path, enhance_contrast=True)
                for i in range(len(cells)):
                    folder_cells_enhanced.append(cells[i])
                    cell_metadata.append((f_path, i)) # file path と そのファイル内でのindex
            
            if not folder_cells_enhanced: continue
            
            # まとめてスコア計算
            scores = self.compute_scores(folder_cells_enhanced)
            
            # 結果保存
            results[name] = {
                'sample_name': name,
                'folder_path': folder_path,
                'total_cells': len(folder_cells_enhanced),
                'cons_rate': scores['cons_rate'],
                'mean_mse': np.mean(scores['mse']),
                'is_wt': name.upper() == 'WT'
            }
            
            # 詳細保存 (Raw画像抽出のために file_path と local_idx を保存)
            for i, (mse, c_score) in enumerate(zip(scores['mse'], scores['cons_score'])):
                f_path, local_idx = cell_metadata[i]
                detailed_results.append({
                    'sample_name': name,
                    'file_path': f_path,   # 原本ファイルのパス
                    'local_idx': local_idx, # そのファイル内でのインデックス
                    'global_idx': i,
                    'mse': mse,
                    'cons_score': c_score
                })
        
        print("\nProcessing complete.")
        if results:
            self.save_and_visualize(results, detailed_results, output_dir, wt_thresholds)

    def save_and_visualize(self, results, detailed_results, output_dir, wt_thresholds):
        df_res = pd.DataFrame.from_dict(results, orient='index')
        df_det = pd.DataFrame(detailed_results)
        
        df_res.to_csv(os.path.join(output_dir, 'summary.csv'))
        df_det.to_csv(os.path.join(output_dir, 'detailed_cell_results.csv'), index=False)
        
        self.plot_anomaly_rates(df_res, output_dir, wt_thresholds)
        self.plot_violin(df_det, output_dir)
        self.generate_phenotype_mosaic(df_res, df_det, output_dir)

    def plot_anomaly_rates(self, df, output_dir, thresholds):
        plt.figure(figsize=(12, 6))
        names = [n[:15] for n in df['sample_name']]
        colors = ['lightblue' if not w else 'blue' for w in df['is_wt']]
        plt.bar(range(len(names)), df['cons_rate']*100, color=colors)
        plt.axhline(thresholds['rate'], color='blue', linestyle='--', label='WT Baseline')
        plt.axhline(thresholds['threshold'], color='red', linestyle='--', label='Threshold')
        plt.xticks(range(len(names)), names, rotation=45, ha='right')
        plt.ylabel('Anomaly Rate (%)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'anomaly_rates.png'), dpi=300)
        plt.close()

    def plot_violin(self, df, output_dir):
        plt.figure(figsize=(14, 8))
        order = ['WT'] + sorted([s for s in df['sample_name'].unique() if s != 'WT'])
        if 'WT' not in df['sample_name'].values: order = sorted(df['sample_name'].unique())
        
        sns.violinplot(x='sample_name', y='cons_score', data=df, order=order, palette='Set2', inner='quartile')
        
        wt_data = df[df['sample_name'].str.upper() == 'WT']
        if not wt_data.empty:
            p99 = wt_data['cons_score'].quantile(0.99)
            plt.axhline(p99, color='red', linestyle='--', label=f'WT 99% ({p99:.2f})')
            plt.legend()
            
        plt.xticks(rotation=45, ha='right')
        plt.title('Anomaly Score Distribution (Folder-Based)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'violin_plot.png'), dpi=300)
        plt.close()

    def generate_phenotype_mosaic(self, df_res, df_det, output_dir, top_n=5):
        print("  Generating Phenotype Mosaic (Raw)...")
        # WT + Top Anomalies
        mutants = df_res[df_res['sample_name'] != 'WT'].sort_values('cons_rate', ascending=False).head(5)
        targets = []
        if 'WT' in df_res['sample_name'].values: targets.append('WT')
        targets.extend(mutants['sample_name'].tolist())
        
        if not targets: return

        n_rows = len(targets)
        n_cols = top_n
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*2, n_rows*2.2))
        if n_rows == 1: axes = np.array([axes])
        if n_cols == 1: axes = np.array([[ax] for ax in axes])
        elif axes.ndim == 1: axes = axes.reshape(n_rows, n_cols)
        
        for r, sample in enumerate(targets):
            s_data = df_det[df_det['sample_name'] == sample]
            
            # 候補選定
            if sample == 'WT':
                candidates = s_data.iloc[(s_data['cons_score'] - 0).abs().argsort()].head(top_n)
                label = "(Typical)"
            else:
                candidates = s_data.sort_values('cons_score', ascending=False).head(top_n)
                label = "(Anomaly)"
            
            # 行タイトル
            axes[r, 0].text(-0.2, 0.5, f"{sample}\n{label}", transform=axes[r, 0].transAxes, 
                           va='center', ha='right', fontsize=11, fontweight='bold')
            
            # 各細胞を描画
            for c in range(n_cols):
                ax = axes[r, c]
                if c < len(candidates):
                    row = candidates.iloc[c]
                    file_path = row['file_path']
                    local_idx = row['local_idx'] # 保存しておいたローカルインデックス
                    score = row['cons_score']
                    
                    # そのファイルを開いて、指定インデックスの細胞だけRaw抽出
                    raw_cells = self.extract_quality_cells(file_path, enhance_contrast=False)
                    
                    if local_idx < len(raw_cells):
                        img = raw_cells[local_idx]
                        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
                        
                        col = 'blue' if sample == 'WT' else ('red' if score > 13 else 'black')
                        ax.set_title(f"{score:.1f}", color=col, fontsize=10, fontweight='bold')
                    else:
                        ax.text(0.5, 0.5, "Idx Err", ha='center')
                ax.axis('off')
        
        plt.suptitle("Phenotype Mosaic (Folder-Based / Raw Images)", fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'phenotype_mosaic_raw.png'), dpi=300)
        plt.close()

def main():
    # ================= SETTINGS =================
    model_dir = "/Users/matsuokoujirou/Documents/Data/Screening/Models/20250613_1524_improved"
    
    # ターゲット: ここに「WTフォルダ」「変異株Aフォルダ」「変異株Bフォルダ」などが並んでいる親フォルダを指定
    root_folder_containing_samples = "/Users/matsuokoujirou/Documents/Data/Screening/250603_check" 
    
    # または手動辞書を使うなら下記をコメントアウト解除して指定
    # manual_folders = {
    #     "WT": "/path/to/WT",
    #     "SampleA": "/path/to/SampleA"
    # }
    # ============================================

    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    output_dir = f"/Users/matsuokoujirou/Documents/Data/Screening/Results/{timestamp}_folder_suite"
    
    screener = ProductionMutantScreeningFolder(model_dir)
    
    # 1. 自動取得モード
    folders_dict = screener.get_subfolders(root_folder_containing_samples)
    
    # 2. 手動モード（自動取得が空、あるいは手動優先の場合）
    # folders_dict = manual_folders 
    
    if folders_dict:
        screener.screen_folders(folders_dict, output_dir)
        print(f"\nCompleted! Results: {output_dir}")
    else:
        print("No folders found.")

if __name__ == "__main__":
    main()