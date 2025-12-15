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

class ProductionMutantScreening:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.load_trained_models()
        
    def load_trained_models(self):
        """訓練済みモデルの読み込み"""
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
        
        # StarDistモデルのロード（失敗時はダウンロード）
        try:
            self.stardist_model = StarDist2D.from_pretrained('2D_versatile_fluo')
        except:
            print("Downloading StarDist model...")
            self.stardist_model = StarDist2D.from_pretrained('2D_versatile_fluo')
            
        print("All models loaded successfully!")

    def get_files_from_folder(self, folder_path):
        """フォルダ内のTIFファイルを自動取得"""
        if not os.path.exists(folder_path):
            print(f"Folder not found: {folder_path}")
            return {}
        
        tif_files = sorted(glob(os.path.join(folder_path, '*.tif')) + glob(os.path.join(folder_path, '*.tiff')))
        # ファイル名（拡張子なし）をキーにする
        files_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in tif_files}
        
        print(f"Found {len(files_dict)} TIF files in {folder_path}")
        return files_dict
    
    def extract_quality_cells(self, image_path, enhance_contrast=True):
        """
        細胞抽出・品質管理
        enhance_contrast=True: 解析用（コントラスト強調あり）- AIに入力するため
        enhance_contrast=False: 表示用（生画像）- モザイク画像生成のため
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
            # 表示正規化用の最大輝度値
            img_max = np.max(green_c) if np.max(green_c) > 0 else 1
            
            for prop in props:
                minr, minc, maxr, maxc = prop.bbox
                # フィルタリング
                if (minr < 10 or minc < 10 or maxr > (h - 10) or maxc > (w - 10)): continue
                if prop.area < 200 or prop.area > 8000: continue
                if prop.eccentricity > 0.95: continue
                
                cell_img = green_c[minr:maxr, minc:maxc]
                # 強度フィルタ
                if np.mean(cell_img) < 0.5 or np.std(cell_img) < 0.1: continue
                
                if enhance_contrast:
                    # 解析用: コントラスト強調 + 正規化 (AI学習時と同じ処理)
                    # ここで最大値正規化を行うかは学習時の処理に合わせる（通常は個別に正規化することが多い）
                    cell_float = cell_img / np.max(cell_img) if np.max(cell_img) > 0 else cell_img
                    cell_eq = exposure.equalize_adapthist(cell_float, clip_limit=0.02)
                    cell_final = resize(cell_eq, (64, 64), anti_aliasing=True)
                else:
                    # 表示用: 生画像 + リサイズのみ
                    # 砂嵐防止のため、個別の正規化や強調はしない
                    cell_resized = resize(cell_img, (64, 64), anti_aliasing=True, preserve_range=True)
                    # 表示用に0-1範囲に収める（画像全体の最大値で割ることで相対的な暗さを維持）
                    cell_final = cell_resized / img_max
                
                cells.append(cell_final)
            
            return cells
            
        except Exception as e:
            print(f"Error extracting cells from {os.path.basename(image_path)}: {e}")
            return []

    def compute_anomaly_scores(self, cell_images):
        if len(cell_images) == 0: return {}
        X = np.expand_dims(np.array(cell_images), axis=-1).astype('float32')
        
        # 再構成誤差
        reconstructed = self.autoencoder.predict(X, verbose=0)
        mse = np.mean(np.square(X - reconstructed), axis=(1, 2, 3))
        
        # 特徴抽出と異常検知
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
            'cons_rate': np.sum(cons_pred == -1) / len(cons_pred),
            'mod_rate': np.sum(mod_pred == -1) / len(mod_pred)
        }

    def screen_samples(self, files_dict, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print("\n=== Starting Automated Screening ===")
        
        results = {}
        detailed_results = []
        
        # 1. WT Baselineの計算
        wt_path = None
        # ファイル名に "WT" が含まれるものを探す（大文字小文字区別なし）
        for name, path in files_dict.items():
            if name.upper() == 'WT': wt_path = path; break
        
        # 見つからない場合は固定パス（環境に合わせて修正可）
        if not wt_path:
            fixed_wt = "/Users/matsuokoujirou/Documents/Data/Screening/Pools/normal/WT.tif"
            if os.path.exists(fixed_wt):
                wt_path = fixed_wt
                print(f"Using fixed WT path: {wt_path}")
            else:
                print("Warning: WT file not found. Using default thresholds.")
        
        wt_thresholds = {'wt_rate': 0.0, 'threshold': 5.0} # デフォルト
        
        if wt_path and os.path.exists(wt_path):
            print(f"Calculating baseline from: {os.path.basename(wt_path)}")
            wt_cells = self.extract_quality_cells(wt_path, enhance_contrast=True)
            if wt_cells:
                wt_scores = self.compute_anomaly_scores(wt_cells)
                wt_rate = wt_scores['cons_rate'] * 100
                wt_thresholds = {'wt_rate': wt_rate, 'threshold': wt_rate + 4.2}
                print(f"  WT Baseline: {wt_rate:.2f}% | Threshold: {wt_thresholds['threshold']:.2f}%")

        # 2. 全ファイルのスクリーニング
        total_files = len(files_dict)
        for idx, (name, path) in enumerate(files_dict.items(), 1):
            print(f"[{idx}/{total_files}] Processing {name}...", end='\r')
            
            # 解析用（強調あり）で抽出
            cells = self.extract_quality_cells(path, enhance_contrast=True)
            if not cells: continue
            
            scores = self.compute_anomaly_scores(cells)
            
            # 結果格納
            results[name] = {
                'sample_name': name,
                'file_path': path,
                'total_cells': len(cells),
                'cons_rate': scores['cons_rate'],
                'mod_rate': scores['mod_rate'],
                'mean_mse': np.mean(scores['mse']),
                'is_wt': name.upper() == 'WT'
            }
            
            # 詳細（細胞ごとのスコア）格納
            for i, (m, c_sc) in enumerate(zip(scores['mse'], scores['cons_score'])):
                detailed_results.append({
                    'sample_name': name,
                    'file_path': path,
                    'cell_id': i,
                    'mse': m,
                    'cons_score': c_sc
                })
        
        print("\nProcessing complete. Generating reports and visualizations...")
        
        if results:
            self.save_and_visualize(results, detailed_results, output_dir, wt_thresholds)
        else:
            print("No valid results to save.")

    def save_and_visualize(self, results, detailed_results, output_dir, wt_thresholds):
        # DataFrame化と保存
        df_res = pd.DataFrame.from_dict(results, orient='index')
        df_det = pd.DataFrame(detailed_results)
        
        df_res.to_csv(os.path.join(output_dir, 'screening_summary.csv'))
        df_det.to_csv(os.path.join(output_dir, 'detailed_cell_results.csv'), index=False)
        
        # 可視化関数の呼び出し
        self.plot_anomaly_rates(df_res, output_dir, wt_thresholds)
        self.plot_mse_distributions(df_res, df_det, output_dir)
        self.plot_violin(df_det, output_dir) # ★Violin Plot (結合済み)
        self.generate_phenotype_mosaic(df_res, df_det, output_dir) # ★Raw Mosaic (結合済み)

    def plot_anomaly_rates(self, df, output_dir, thresholds):
        """異常率の棒グラフ"""
        plt.figure(figsize=(14, 7))
        # 名前が長い場合は省略
        names = [n[:15] + '..' if len(n)>15 else n for n in df['sample_name']]
        
        # 異常率が高い順、またはアルファベット順などでソートしたければここでdfをソート
        # ここではファイル読み込み順（通常アルファベット順）のまま
        
        colors = ['lightblue' if not is_wt else 'blue' for is_wt in df['is_wt']]
        plt.bar(range(len(names)), df['cons_rate']*100, color=colors, alpha=0.7)
        
        plt.axhline(thresholds['wt_rate'], color='blue', linestyle='--', label=f"WT Baseline ({thresholds['wt_rate']:.1f}%)")
        plt.axhline(thresholds['threshold'], color='red', linestyle='--', label=f"Threshold ({thresholds['threshold']:.1f}%)")
        
        plt.xticks(range(len(names)), names, rotation=45, ha='right', fontsize=10)
        plt.ylabel('Anomaly Rate (%)')
        plt.title('Anomaly Rates by Sample')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'anomaly_rates.png'), dpi=300)
        plt.close()

    def plot_mse_distributions(self, df_res, df_det, output_dir):
        """MSE分布図 (Top 8 Anomaly + WT)"""
        top_anomalies = df_res.sort_values('cons_rate', ascending=False).head(8)
        targets = top_anomalies['sample_name'].tolist()
        if 'WT' not in targets and 'WT' in df_res['sample_name'].values:
            targets.append('WT')
            
        n_plots = len(targets)
        cols = 3
        rows = (n_plots + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3))
        if rows == 1: axes = np.array([axes])
        if cols == 1: axes = np.array([[ax] for ax in axes])
        axes = axes.flatten()
        
        for i, sample in enumerate(targets):
            ax = axes[i]
            data = df_det[df_det['sample_name'] == sample]
            is_wt = (sample.upper() == 'WT')
            color = 'blue' if is_wt else 'red'
            
            ax.hist(data['mse'], bins=30, color=color, alpha=0.6, density=True)
            ax.set_title(sample)
            ax.set_xlabel('MSE')
            
        for j in range(i+1, len(axes)): axes[j].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'mse_distributions.png'), dpi=300)
        plt.close()

    def plot_violin(self, df_det, output_dir):
        """★Violin Plot (ご要望の結合部分)★"""
        print("  Generating Violin Plot...")
        plt.figure(figsize=(14, 8))
        
        # WTを先頭にし、他はアルファベット順
        samples = df_det['sample_name'].unique().tolist()
        order = ['WT'] + sorted([s for s in samples if s != 'WT'])
        if 'WT' not in samples: order = sorted(samples)
        
        sns.violinplot(
            x='sample_name', 
            y='cons_score', 
            data=df_det, 
            order=order, 
            palette='Set2', 
            inner='quartile' # 四分位線を表示
        )
        
        # WTの99%タイルラインの描画
        wt_data = df_det[df_det['sample_name'].str.upper() == 'WT']
        if not wt_data.empty:
            p99 = wt_data['cons_score'].quantile(0.99)
            plt.axhline(p99, color='red', linestyle='--', linewidth=2, 
                       label=f'WT 99th Percentile ({p99:.2f})')
            plt.legend(loc='upper right')
            
        plt.title('Anomaly Score Distribution (Violin Plot)', fontsize=16)
        plt.ylabel('Anomaly Score (Higher = More Abnormal)', fontsize=12)
        plt.xlabel('Sample', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'violin_plot.png'), dpi=300)
        plt.close()

    def generate_phenotype_mosaic(self, df_res, df_det, output_dir, top_n_samples=5, top_n_cells=5):
        """★Phenotype Mosaic (RAW画像版)★"""
        print("  Generating Phenotype Mosaic (Raw Images)...")
        
        # ターゲット選定: 異常率が高い順TopN + WT
        # WTを除外してソート
        mutants = df_res[df_res['sample_name'] != 'WT'].sort_values('cons_rate', ascending=False)
        top_mutants = mutants.head(top_n_samples)['sample_name'].tolist()
        
        targets = []
        if 'WT' in df_res['sample_name'].values: targets.append('WT')
        targets.extend(top_mutants)
        
        if not targets: return

        # プロット準備
        n_rows = len(targets)
        n_cols = top_n_cells
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*2, n_rows*2.2))
        
        # 軸の正規化
        if n_rows == 1: axes = np.array([axes])
        if n_cols == 1: axes = np.array([[ax] for ax in axes])
        elif axes.ndim == 1: axes = axes.reshape(n_rows, n_cols)
        
        for r, sample in enumerate(targets):
            # サンプルのファイルパスと詳細データ
            s_info = df_res.loc[sample]
            file_path = s_info['file_path']
            s_data = df_det[df_det['sample_name'] == sample]
            
            if s_data.empty: continue
            
            # 細胞選定 (WTは正常、変異株は異常)
            if sample == 'WT':
                # スコアの絶対値が小さい（0に近い）順
                candidates = s_data.iloc[(s_data['cons_score'] - 0).abs().argsort()].head(top_n_cells)
                suffix = "(Typical)"
            else:
                # スコアが大きい順
                candidates = s_data.sort_values('cons_score', ascending=False).head(top_n_cells)
                suffix = "(Anomaly)"
            
            target_ids = candidates['cell_id'].tolist()
            scores = candidates['cons_score'].tolist()
            
            # ★重要★ 生画像の再抽出 (enhance_contrast=False)
            # 注意: file_basedなので、extract_quality_cellsで返るリストのインデックスとcell_idは一致する
            raw_cells = self.extract_quality_cells(file_path, enhance_contrast=False)
            
            # 行ラベル
            axes[r, 0].text(-0.2, 0.5, f"{sample}\n{suffix}", 
                           transform=axes[r, 0].transAxes, va='center', ha='right', fontsize=11, fontweight='bold')
            
            for c in range(n_cols):
                ax = axes[r, c]
                if c < len(target_ids):
                    cid = target_ids[c]
                    score = scores[c]
                    
                    if cid < len(raw_cells):
                        img = raw_cells[cid]
                        # 生画像を表示 (0-1範囲)
                        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
                        
                        # スコア表示 (WT基準ライン超えは赤)
                        # 閾値はグラフ描画時の99%ラインなどを参照したいが、ここでは簡易的に13.0とするか、
                        # Violin Plotで計算した値を渡すのがベスト。今回は固定値または相対値で色付け。
                        text_color = 'red' if score > 13.0 else 'black'
                        if sample == 'WT': text_color = 'blue'
                        
                        ax.set_title(f"{score:.1f}", color=text_color, fontsize=10, fontweight='bold')
                    else:
                        ax.text(0.5, 0.5, "Err", ha='center')
                ax.axis('off')
                
        plt.suptitle("Phenotype Mosaic (Raw Images - No Contrast Enhancement)", fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'phenotype_mosaic_raw.png'), dpi=300)
        plt.close()


def main():
    # ================= SETTINGS =================
    # モデルのディレクトリ
    model_dir = "/Users/matsuokoujirou/Documents/Data/Screening/Models/20250613_1524_improved"
    
    # 解析したいフォルダパス (ここにTIFファイルが入っているフォルダを指定)
    target_folder = "/Users/matsuokoujirou/Documents/Data/Screening/Candidates/251127"
    # ============================================

    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    output_dir = f"/Users/matsuokoujirou/Documents/Data/Screening/Results/{timestamp}_auto_screening"
    
    screener = ProductionMutantScreening(model_dir)
    
    # フォルダからファイルを自動取得
    files_dict = screener.get_files_from_folder(target_folder)
    
    if files_dict:
        # スクリーニング実行 (可視化も含む)
        screener.screen_samples(files_dict, output_dir)
        print(f"\nAll Done! Results saved to:\n{output_dir}")
    else:
        print("No files found to process.")

if __name__ == "__main__":
    main()