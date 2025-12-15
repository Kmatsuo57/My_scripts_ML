import numpy as np
import tifffile as tiff
import os
import os.path
from glob import glob
from tensorflow.keras.models import load_model
from stardist.models import StarDist2D
from csbdeep.utils import normalize
from skimage.measure import regionprops
from skimage.segmentation import clear_border
from skimage.transform import resize
from skimage import exposure
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime
import pickle
import umap.umap_ as umap
from mpl_toolkits.mplot3d import Axes3D

# StarDistのラベル関数はskimage.measureからインポート
from skimage.measure import label as skimage_label


class ProductionMutantScreening:
    """
    訓練済みモデル（CAE, PCA, OC-SVM）を使用して、変異株サンプルの画像から
    細胞を抽出、異常スコアを計算し、UMAP可視化まで行うプロダクションパイプライン。
    """
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.load_trained_models()
        
    def get_files_from_folder(self, folder_path):
        """フォルダ内の全TIFファイルを取得してファイルベースの辞書を作成"""
        if not os.path.exists(folder_path):
            print(f"Folder not found: {folder_path}")
            return {}
        
        tif_files = sorted(glob(os.path.join(folder_path, '*.tif')) + glob(os.path.join(folder_path, '*.tiff')))
        if not tif_files:
            print(f"No TIF/TIFF files found in {folder_path}")
            return {}
        
        files_dict = {}
        for file_path in tif_files:
            # 修正: os.splitext -> os.path.splitext
            sample_name = os.path.splitext(os.path.basename(file_path))[0]
            files_dict[sample_name] = file_path
        
        print(f"Found {len(tif_files)} TIF/TIFF files in {folder_path}")
        return files_dict
        
    def load_trained_models(self):
        print("Loading trained models...")
        try:
            self.autoencoder = load_model(os.path.join(self.model_dir, 'best_autoencoder.keras'), compile=False)
            self.encoder = load_model(os.path.join(self.model_dir, 'encoder.keras'), compile=False)
            
            with open(os.path.join(self.model_dir, 'scaler.pkl'), 'rb') as f:
                self.scaler = pickle.load(f)
            with open(os.path.join(self.model_dir, 'pca.pkl'), 'rb') as f:
                self.pca = pickle.load(f)
            
            with open(os.path.join(self.model_dir, 'detector_conservative.pkl'), 'rb') as f:
                self.detector_conservative = pickle.load(f)
            with open(os.path.join(self.model_dir, 'detector_moderate.pkl'), 'rb') as f:
                self.detector_moderate = pickle.load(f)
            
            self.stardist_model = StarDist2D.from_pretrained('2D_versatile_fluo')
            print("All models loaded successfully!")
        except Exception as e:
            print(f"[ERROR] Failed to load models: {e}")
            raise
    
    def extract_quality_cells(self, image_path):
        try:
            image = tiff.imread(image_path)
            if image.ndim == 3 and image.shape[-1] >= 3:
                green_channel = image[..., 1]
                seg_channel = image[..., 2]
            elif image.ndim == 2:
                green_channel = image
                seg_channel = image
            else:
                green_channel = image[0] if image.ndim == 3 else image
                seg_channel = image[0] if image.ndim == 3 else image
            
            normalized_seg = normalize(seg_channel)
            labels, details = self.stardist_model.predict_instances(normalized_seg)
            
            height, width = labels.shape
            props = regionprops(labels)
            quality_cells = []
            
            for prop in props:
                minr, minc, maxr, maxc = prop.bbox
                if (minr < 10 or minc < 10 or maxr > (height - 10) or maxc > (width - 10)): continue
                if prop.area < 200 or prop.area > 8000: continue
                if prop.eccentricity > 0.95: continue
                
                cell_image = green_channel[minr:maxr, minc:maxc]
                cell_mean = np.mean(cell_image)
                cell_std = np.std(cell_image)
                if cell_mean < 0.5 or cell_std < 0.1: continue
                
                cell_image_float = cell_image / np.max(green_channel) if np.max(green_channel) > 0 else cell_image
                cell_image_eq = exposure.equalize_adapthist(cell_image_float, clip_limit=0.02)
                cell_image_resized = resize(cell_image_eq, (64, 64), anti_aliasing=True)
                quality_cells.append(cell_image_resized)
            
            return quality_cells
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return []
    
    def compute_anomaly_scores(self, cell_images):
        if len(cell_images) == 0: return None
        
        X = np.expand_dims(np.array(cell_images), axis=-1).astype('float32')
        reconstructed = self.autoencoder.predict(X, verbose=0)
        mse_errors = np.mean(np.square(X - reconstructed), axis=(1, 2, 3))
        mae_errors = np.mean(np.abs(X - reconstructed), axis=(1, 2, 3))
        
        encoded_features = self.encoder.predict(X, verbose=0)
        encoded_flat = encoded_features.reshape(len(encoded_features), -1)
        encoded_scaled = self.scaler.transform(encoded_flat)
        encoded_pca = self.pca.transform(encoded_scaled)
        
        conservative_predictions = self.detector_conservative.predict(encoded_pca)
        moderate_predictions = self.detector_moderate.predict(encoded_pca)
        
        return {
            'reconstruction_mse': mse_errors,
            'conservative_predictions': conservative_predictions,
            'moderate_predictions': moderate_predictions,
            'conservative_anomaly_rate': np.sum(conservative_predictions == -1) / len(conservative_predictions),
            'moderate_anomaly_rate': np.sum(moderate_predictions == -1) / len(moderate_predictions),
            'encoded_pca': encoded_pca
        }
    
    def screen_mutant_samples_by_folder(self, test_folders_dict, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print("=== Starting Mutant Screening (Folder-based) ===")
        
        results = {}
        detailed_results = []
        
        # WT基準計算
        wt_path = test_folders_dict.get('WT')
        if not wt_path: raise ValueError("WT path required")
        
        # 全データ収集用
        all_features_data = {'features': [], 'sample_name': [], 'is_anomaly': [], 'mse': []}

        for folder_name, folder_path in test_folders_dict.items():
            print(f"\nProcessing folder: {folder_name}...")
            if not os.path.exists(folder_path):
                print(f"  Folder not found: {folder_path}")
                continue
                
            tif_files = sorted(glob(os.path.join(folder_path, '*.tif')) + glob(os.path.join(folder_path, '*.tiff')))
            all_cells = []
            
            for file_path in tif_files:
                cells = self.extract_quality_cells(file_path)
                all_cells.extend(cells)
            
            print(f"  Total cells: {len(all_cells)}")
            if len(all_cells) == 0: continue
            
            scores = self.compute_anomaly_scores(all_cells)
            
            # 結果保存
            results[folder_name] = {
                'sample_name': folder_name,
                'total_cells': len(all_cells),
                'conservative_anomaly_rate': scores['conservative_anomaly_rate'],
                'mean_mse': np.mean(scores['reconstruction_mse'])
            }
            
            # UMAP用データ収集
            all_features_data['features'].append(scores['encoded_pca'])
            all_features_data['sample_name'].extend([folder_name] * len(all_cells))
            all_features_data['is_anomaly'].extend(scores['conservative_predictions'] == -1)
            all_features_data['mse'].extend(scores['reconstruction_mse'])
            
            print(f"    Anomaly rate: {scores['conservative_anomaly_rate']*100:.2f}%")

        # WTの異常率から閾値を簡易計算
        if 'WT' in results:
            wt_rate = results['WT']['conservative_anomaly_rate'] * 100
            wt_thresholds = {
                'wt_conservative_rate': wt_rate,
                'threshold_conservative': wt_rate + 4.2
            }
        else:
            wt_thresholds = {'wt_conservative_rate': 0, 'threshold_conservative': 0}

        # 保存と可視化
        self.save_and_visualize_results(results, output_dir, wt_thresholds, all_features_data)
        return results

    def save_and_visualize_results(self, results, output_dir, wt_thresholds, all_features_data):
        # CSV保存
        pd.DataFrame.from_dict(results, orient='index').to_csv(os.path.join(output_dir, 'summary.csv'))
        
        # UMAP可視化
        self.create_umap_visualization(all_features_data, output_dir)

    def create_umap_visualization(self, all_features_data, output_dir):
        print("\n=== Generating UMAP Visualization ===")
        X_features = np.concatenate(all_features_data['features'], axis=0)
        
        umap_df = pd.DataFrame({
            'sample': all_features_data['sample_name'],
            'is_anomaly': all_features_data['is_anomaly'],
            'mse': all_features_data['mse']
        })
        
        # UMAP計算（2D & 3D）
        print("  Computing 2D UMAP...")
        reducer_2d = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        embedding_2d = reducer_2d.fit_transform(X_features)
        umap_df['UMAP1'] = embedding_2d[:, 0]
        umap_df['UMAP2'] = embedding_2d[:, 1]
        
        print("  Computing 3D UMAP...")
        reducer_3d = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=3, random_state=42)
        embedding_3d = reducer_3d.fit_transform(X_features)
        umap_df['UMAP3_1'] = embedding_3d[:, 0]
        umap_df['UMAP3_2'] = embedding_3d[:, 1]
        umap_df['UMAP3_3'] = embedding_3d[:, 2]

        # 色設定
        samples = sorted(umap_df['sample'].unique())
        palette = sns.color_palette("tab10", len(samples)) # 視認性の高いパレット
        color_map = dict(zip(samples, palette))

        # ---------------------------------------------------------
        # 1. 全サンプル 2D (凡例は変異株名のみ)
        # ---------------------------------------------------------
        plt.figure(figsize=(10, 8))
        # is_anomalyによるサイズ変更は残すが、凡例からは除外するために工夫
        sns.scatterplot(
            x='UMAP1', y='UMAP2', hue='sample', style='sample',
            size='is_anomaly', sizes=(10, 40), alpha=0.7,
            data=umap_df, palette=color_map
        )
        # 凡例を整理（タイトル変更、is_anomaly除外）
        handles, labels = plt.gca().get_legend_handles_labels()
        # サンプル名の部分だけ抽出（is_anomalyのTrue/Falseなどが来る前まで）
        # 簡易的にサンプル数分だけ取得
        custom_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[s], markersize=10) for s in samples]
        plt.legend(custom_handles, samples, title='Strain', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.title('UMAP 2D: All Samples')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'umap_2d_all_clean_legend.png'), dpi=300)
        plt.close() # 表示せずに閉じる

        # ---------------------------------------------------------
        # 2. 全サンプル 3D (ファイル保存)
        # ---------------------------------------------------------
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        for sample_name in samples:
            subset = umap_df[umap_df['sample'] == sample_name]
            ax.scatter(
                subset['UMAP3_1'], subset['UMAP3_2'], subset['UMAP3_3'],
                c=[color_map[sample_name]], label=sample_name,
                s=subset['is_anomaly'].apply(lambda x: 30 if x else 5), # 異常は大きく
                alpha=0.6
            )
        
        ax.set_xlabel('UMAP1')
        ax.set_ylabel('UMAP2')
        ax.set_zlabel('UMAP3')
        ax.legend(title='Strain', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title('UMAP 3D: All Samples')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'umap_3d_all.png'), dpi=300)
        plt.close()

        # ---------------------------------------------------------
        # 3. 個別強調プロット (WT(色付き) vs 1系列(色付き) vs その他(グレー))
        # ---------------------------------------------------------
        print("  Generating individual highlight plots (WT colored)...")
        n_plots = len(samples)
        cols = 3
        rows = (n_plots + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        axes = axes.flatten()
        
        # WTデータを取得
        wt_data = umap_df[umap_df['sample'] == 'WT']
        wt_color = color_map.get('WT', 'blue') # WTの色を取得、なければ青
        
        bg_color = '#E0E0E0' # その他のデータの薄いグレー
        bg_alpha = 0.1
        
        for i, target_sample in enumerate(samples):
            ax = axes[i]
            
            # 1. 背景（ターゲットでもWTでもないデータ）をグレーでプロット
            other_data = umap_df[(umap_df['sample'] != target_sample) & (umap_df['sample'] != 'WT')]
            ax.scatter(
                other_data['UMAP1'], other_data['UMAP2'],
                c=bg_color, s=5, alpha=bg_alpha, label='Others'
            )
            
            # 2. WTを色付きでプロット（ターゲットがWTでない場合）
            if target_sample != 'WT':
                ax.scatter(
                    wt_data['UMAP1'], wt_data['UMAP2'],
                    c=[wt_color], s=5, alpha=0.3, label='WT'
                )
            
            # 3. ターゲット変異株を色付きで前面にプロット
            target_data = umap_df[umap_df['sample'] == target_sample]
            # ターゲットがWTの場合は既にプロット済みだが、強調のため再描画しても良い
            ax.scatter(
                target_data['UMAP1'], target_data['UMAP2'],
                c=[color_map[target_sample]], s=15, alpha=0.8, label=target_sample
            )
            
            ax.set_title(target_sample, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            
            # 凡例の作成
            if target_sample != 'WT':
                # カスタム凡例: ターゲットとWTのみ表示
                legend_elements = [
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[target_sample], label=target_sample),
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=wt_color, label='WT', alpha=0.5)
                ]
                ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
            else:
                 # ターゲットがWTの場合
                legend_elements = [
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=wt_color, label='WT')
                ]
                ax.legend(handles=legend_elements, loc='upper right', fontsize=8)


        # 余ったサブプロットを非表示
        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)
            
        plt.suptitle('UMAP Individual Highlights: Mutant vs WT (Colored) vs Others (Gray)', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(output_dir, 'umap_2d_individual_highlights_wt_colored.png'), dpi=300)
        plt.close()
        
        print("Visualization completed.")

def main():
    # モデルパス（実際のパスに合わせてください）
    model_dir = "/Users/matsuokoujirou/Documents/Data/Screening/Models/20250613_1524_improved"
    
    # データパス
    test_folders = {
        "WT": "/Users/matsuokoujirou/Documents/Data/Screening/Noemal_cells/images",
        'Candidate1': "/Users/matsuokoujirou/Documents/Data/Screening/Candidates/Candidate1",
        'Candidate2': "/Users/matsuokoujirou/Documents/Data/Screening/Candidates/Candidate2",
        'Candidate3': "/Users/matsuokoujirou/Documents/Data/Screening/Candidates/Candidate3",
    }
    
    output_dir = f"./screening_results/{datetime.now().strftime('%Y%m%d_%H%M')}_folder_based_screening"
    
    try:
        screener = ProductionMutantScreening(model_dir)
        if not test_folders: return
        
        screener.screen_mutant_samples_by_folder(test_folders, output_dir)
        print(f"\n=== SCREENING COMPLETED ===\nResults saved to: {output_dir}")

    except Exception as e:
        print(f"\n[CRITICAL ERROR] {e}")

if __name__ == "__main__":
    main()