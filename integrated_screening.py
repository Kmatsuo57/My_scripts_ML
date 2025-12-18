# coding: utf-8
import argparse
import os
import pickle
from datetime import datetime
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tifffile as tiff
import umap.umap_ as umap
from csbdeep.utils import normalize
from mpl_toolkits.mplot3d import Axes3D
from skimage import exposure
from skimage.measure import regionprops
from skimage.transform import resize
from stardist.models import StarDist2D
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, MaxPooling2D

# --- New imports for extended analysis ---
from sklearn.manifold import TSNE
from scipy.stats import wasserstein_distance
try:
    import phate
except ImportError:
    phate = None
try:
    import hdbscan
except ImportError:
    hdbscan = None


class MutantScreeningPipeline:
    """
    変異株スクリーニングの統合パイプライン。
    - ファイル単位での解析 (file mode)
    - フォルダ単位での解析 (folder mode)
    - UMAP可視化 (umap mode)
    - 追加の高度な解析 (extra_viz, quantitative flags)
    をサポートします。
    """

    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.load_trained_models()

    def load_trained_models(self):
        """訓練済みAIモデルと関連ファイルを読み込む"""
        print("Loading trained models...")
        try:
            self.autoencoder = load_model(os.path.join(self.model_dir, 'best_autoencoder.keras'), compile=False)
            
            # --- Dynamically extract Decoder ---
            # 1. Find the last MaxPooling2D layer (output of encoder)
            maxpool_idx = -1
            for i, layer in enumerate(self.autoencoder.layers):
                if isinstance(layer, MaxPooling2D):
                    maxpool_idx = i
            
            if maxpool_idx != -1:
                # 2. Extract decoder layers (everything after the last MaxPooling2D)
                decoder_layers = self.autoencoder.layers[maxpool_idx+1:]
                
                # 3. Construct the decoder model
                # Input shape matches the encoder output (8, 8, 32)
                decoder_input = Input(shape=(8, 8, 32))
                x = decoder_input
                for layer in decoder_layers:
                    x = layer(x)
                self.decoder = Model(decoder_input, x)
                print("  Decoder extracted and reconstructed successfully.")
            else:
                print("[WARNING] MaxPooling2D layer not found. Decoder extraction skipped.")

            self.encoder = load_model(os.path.join(self.model_dir, 'encoder.keras'), compile=False)
            with open(os.path.join(self.model_dir, 'scaler.pkl'), 'rb') as f:
                self.scaler = pickle.load(f)
            with open(os.path.join(self.model_dir, 'pca.pkl'), 'rb') as f:
                self.pca = pickle.load(f)
            with open(os.path.join(self.model_dir, 'detector_conservative.pkl'), 'rb') as f:
                self.detector_conservative = pickle.load(f)
            self.stardist_model = StarDist2D.from_pretrained('2D_versatile_fluo')
            print("All models loaded successfully!")
        except Exception as e:
            print(f"[ERROR] Failed to load models from {self.model_dir}: {e}")
            raise

    def extract_quality_cells(self, image_path, enhance_contrast=True):
        """単一のTIF画像から品質基準を満たす細胞画像を抽出する
        Returns:
            list of tuples: [(raw_cell, preprocessed_cell), ...]
        """
        try:
            image = tiff.imread(image_path)
            if image.ndim == 3 and image.shape[-1] >= 3:
                green_channel, seg_channel = image[..., 1], image[..., 2]
            else:
                green_channel = seg_channel = image
            normalized_seg = normalize(seg_channel)
            labels, _ = self.stardist_model.predict_instances(normalized_seg)
            props = regionprops(labels)
            quality_cells = []
            height, width = labels.shape
            img_max_for_display = np.max(green_channel) if np.max(green_channel) > 0 else 1.0
            for prop in props:
                minr, minc, maxr, maxc = prop.bbox
                if minr < 10 or minc < 10 or maxr > (height - 10) or maxc > (width - 10): continue
                if not (200 <= prop.area <= 8000): continue
                if prop.eccentricity > 0.95: continue
                cell_image = green_channel[minr:maxr, minc:maxc]
                if np.mean(cell_image) < 0.5 or np.std(cell_image) < 0.1: continue
                
                # Create raw cell (0-1 normalized, resized, no contrast enhancement)
                raw_resized = resize(cell_image, (64, 64), anti_aliasing=True, preserve_range=True)
                raw_final = raw_resized / img_max_for_display
                
                # Create preprocessed cell (Contrast enhanced if requested)
                if enhance_contrast:
                    cell_float = cell_image / np.max(cell_image) if np.max(cell_image) > 0 else cell_image
                    cell_eq = exposure.equalize_adapthist(cell_float, clip_limit=0.02)
                    cell_final = resize(cell_eq, (64, 64), anti_aliasing=True)
                else:
                    cell_final = raw_final
                    
                quality_cells.append((raw_final, cell_final))
            return quality_cells
        except Exception as e:
            print(f"Error extracting cells from {os.path.basename(image_path)}: {e}")
            return []

    def compute_anomaly_scores(self, cell_images):
        """細胞画像のリストから異常関連スコアを計算する"""
        if len(cell_images) == 0: return {}
        X = np.expand_dims(np.array(cell_images), axis=-1).astype('float32')
        reconstructed = self.autoencoder.predict(X, verbose=0)
        mse = np.mean(np.square(X - reconstructed), axis=(1, 2, 3))
        encoded_features = self.encoder.predict(X, verbose=0)
        encoded_flat = encoded_features.reshape(len(encoded_features), -1)
        encoded_scaled = self.scaler.transform(encoded_flat)
        encoded_pca = self.pca.transform(encoded_scaled)
        predictions = self.detector_conservative.predict(encoded_pca)
        anomaly_scores = -self.detector_conservative.decision_function(encoded_pca)
        return {
            'mse': mse, 'predictions': predictions, 'anomaly_scores': anomaly_scores,
            'anomaly_rate': np.sum(predictions == -1) / len(predictions),
            'features_pca': encoded_pca
        }

    def _get_files_from_paths(self, input_paths):
        files_dict = {}
        for path in input_paths:
            if os.path.isdir(path):
                tif_files = sorted(glob(os.path.join(path, '*.tif')) + glob(os.path.join(path, '*.tiff')))
                for f in tif_files:
                    files_dict[os.path.splitext(os.path.basename(f))[0]] = f
            elif os.path.isfile(path) and (path.endswith('.tif') or path.endswith('.tiff')):
                files_dict[os.path.splitext(os.path.basename(path))[0]] = path
        return files_dict

    def _get_folders_from_path(self, root_path):
        if not os.path.isdir(root_path): return {}
        subfolders = [f.path for f in os.scandir(root_path) if f.is_dir()]
        return {os.path.basename(f): f for f in subfolders}

    def _calculate_wt_baseline(self, wt_path):
        print(f"Calculating baseline from WT: {wt_path}")
        wt_cells = []
        if os.path.isdir(wt_path):
            tif_files = sorted(glob(os.path.join(wt_path, '*.tif')) + glob(os.path.join(wt_path, '*.tiff')))
            for f in tif_files:
                # Unpack tuples and keep only preprocessed cells for scoring
                cells_data = self.extract_quality_cells(f, enhance_contrast=True)
                wt_cells.extend([c[1] for c in cells_data])
        elif os.path.isfile(wt_path):
            cells_data = self.extract_quality_cells(wt_path, enhance_contrast=True)
            wt_cells = [c[1] for c in cells_data]
        if not wt_cells:
            print("Warning: No cells found in WT. Using default threshold.")
            return {'wt_rate': 0.0, 'threshold': 5.0, 'p99_score': 13.0}
        wt_scores = self.compute_anomaly_scores(wt_cells)
        wt_rate = wt_scores['anomaly_rate'] * 100
        p99_score = np.quantile(wt_scores['anomaly_scores'], 0.99)
        thresholds = {'wt_rate': wt_rate, 'threshold': wt_rate + 4.2, 'p99_score': p99_score}
        print(f"  WT Baseline Rate: {wt_rate:.2f}% | Threshold: {thresholds['threshold']:.2f}% | 99th Score: {p99_score:.2f}")
        return thresholds

    def run_file_mode(self, input_paths, output_dir, wt_path=None):
        print("\n=== Running in FILE mode ===")
        files_dict = self._get_files_from_paths(input_paths)
        if not files_dict:
            print("No TIF files found.")
            return
        os.makedirs(output_dir, exist_ok=True)
        wt_thresholds = self._calculate_wt_baseline(wt_path) if wt_path else {'wt_rate': 0.0, 'threshold': 5.0, 'p99_score': 13.0}
        summary_results, detailed_results = {}, []
        for name, path in files_dict.items():
            print(f"  Processing {name}...")
            cells_data = self.extract_quality_cells(path, enhance_contrast=True)
            if not cells_data: continue
            
            # Unpack: we use preprocessed for scoring
            preprocessed_cells = [c[1] for c in cells_data]
            
            scores = self.compute_anomaly_scores(preprocessed_cells)
            summary_results[name] = {'sample_name': name, 'file_path': path, 'total_cells': len(preprocessed_cells), 'anomaly_rate': scores['anomaly_rate'], 'mean_mse': np.mean(scores['mse']), 'is_wt': 'WT' in name.upper()}
            for i, (score, mse) in enumerate(zip(scores['anomaly_scores'], scores['mse'])):
                detailed_results.append({'sample_name': name, 'file_path': path, 'cell_id': i, 'anomaly_score': score, 'mse': mse})
        if not summary_results:
            print("No results to save.")
            return
        df_summary = pd.DataFrame.from_dict(summary_results, orient='index')
        df_detailed = pd.DataFrame(detailed_results)
        df_summary.to_csv(os.path.join(output_dir, 'summary_file_mode.csv'))
        df_detailed.to_csv(os.path.join(output_dir, 'detailed_results_file_mode.csv'), index=False)
        self.plot_anomaly_rates(df_summary, output_dir, wt_thresholds, "File")
        self.plot_violin(df_detailed, output_dir, wt_thresholds, "File")
        self.generate_phenotype_mosaic(df_summary, df_detailed, output_dir, wt_thresholds, mode='file')
        print(f"File mode processing complete. Results are in {output_dir}")

    def run_folder_mode(self, root_path, output_dir, generate_umap, run_extra_viz, run_quantitative, wt_path=None):
        print(f"\n=== Running in FOLDER mode (UMAP: {generate_umap}, ExtraViz: {run_extra_viz}, Quantitative: {run_quantitative}) ===")
        folders_dict = self._get_folders_from_path(root_path)
        if not folders_dict:
            print("No subfolders found.")
            return
        os.makedirs(output_dir, exist_ok=True)
        if not wt_path and 'WT' in folders_dict:
            wt_path = folders_dict['WT']
        wt_thresholds = self._calculate_wt_baseline(wt_path) if wt_path else {'wt_rate': 0.0, 'threshold': 5.0, 'p99_score': 13.0}
        summary_results, detailed_results, analysis_data = {}, [], {'features': [], 'sample_name': [], 'is_anomaly': [], 'mse': []}
        for name, folder_path in folders_dict.items():
            print(f"  Processing folder: {name}...")
            tif_files = sorted(glob(os.path.join(folder_path, '*.tif')) + glob(os.path.join(folder_path, '*.tiff')))
            if not tif_files: continue
            folder_cells, cell_metadata = [], []
            for f_path in tif_files:
                # Unpack tuple: (raw, preprocessed)
                cells_data = self.extract_quality_cells(f_path, enhance_contrast=True)
                for i, (raw_c, pre_c) in enumerate(cells_data):
                    folder_cells.append(pre_c)
                    cell_metadata.append((f_path, i))
            if not folder_cells: continue
            scores = self.compute_anomaly_scores(folder_cells)
            summary_results[name] = {'sample_name': name, 'folder_path': folder_path, 'total_cells': len(folder_cells), 'anomaly_rate': scores['anomaly_rate'], 'mean_mse': np.mean(scores['mse']), 'is_wt': 'WT' in name.upper()}
            for i, (score, mse) in enumerate(zip(scores['anomaly_scores'], scores['mse'])):
                f_path, local_idx = cell_metadata[i]
                detailed_results.append({'sample_name': name, 'file_path': f_path, 'local_idx': local_idx, 'anomaly_score': score, 'mse': mse})
            if generate_umap or run_extra_viz or run_quantitative:
                analysis_data['features'].append(scores['features_pca'])
                analysis_data['sample_name'].extend([name] * len(folder_cells))
                analysis_data['is_anomaly'].extend(scores['predictions'] == -1)
                analysis_data['mse'].extend(scores['mse'])
        if not summary_results:
            print("No results to save.")
            return
        df_summary = pd.DataFrame.from_dict(summary_results, orient='index')
        df_detailed = pd.DataFrame(detailed_results)
        df_summary.to_csv(os.path.join(output_dir, 'summary_folder_mode.csv'))
        df_detailed.to_csv(os.path.join(output_dir, 'detailed_results_folder_mode.csv'), index=False)
        self.plot_anomaly_rates(df_summary, output_dir, wt_thresholds, "Folder")
        self.plot_violin(df_detailed, output_dir, wt_thresholds, "Folder")
        self.generate_phenotype_mosaic(df_summary, df_detailed, output_dir, wt_thresholds, mode='folder')
        
        # --- Run XAI Analysis (New Logic) ---
        print("  Running XAI analysis (WT Reference & Top 2 Candidates)...")
        
        # 1. WT Reference (Median MSE)
        wt_name = next((s for s in summary_results.keys() if 'WT' in s.upper()), None)
        if wt_name:
            wt_rows = df_detailed[df_detailed['sample_name'] == wt_name]
            if not wt_rows.empty:
                # Use MSE median for reference
                median_row = wt_rows.sort_values('mse').iloc[len(wt_rows)//2]
                try:
                    w_path = median_row['file_path']
                    w_idx = int(median_row['local_idx']) if 'local_idx' in median_row else int(median_row['cell_id'])
                    w_data = self.extract_quality_cells(w_path, enhance_contrast=True)
                    if w_idx < len(w_data):
                        w_raw, w_pre = w_data[w_idx]
                        self.visualize_residuals(w_raw, w_pre, os.path.join(output_dir, 'Reference_WT_Median_residuals.png'))
                        self.visualize_heatmap_overlay(w_raw, w_pre, os.path.join(output_dir, 'Reference_WT_Median_heatmap.png'))
                except Exception as e:
                    print(f"    [Warning] Failed to generate WT Reference: {e}")

        # 2. Top 2 Candidates per Series
        for sample in df_detailed['sample_name'].unique():
            s_rows = df_detailed[df_detailed['sample_name'] == sample]
            # Top 2 by Anomaly Score
            top_2 = s_rows.nlargest(2, 'anomaly_score')
            
            safe_sample_name = "".join(c for c in sample if c.isalnum() or c in ('-', '_')).rstrip()
            
            for rank, (_, row) in enumerate(top_2.iterrows()):
                try:
                    c_path = row['file_path']
                    c_idx = int(row['local_idx']) if 'local_idx' in row else int(row['cell_id'])
                    c_data = self.extract_quality_cells(c_path, enhance_contrast=True)
                    
                    if c_idx < len(c_data):
                        c_raw, c_pre = c_data[c_idx]
                        base_name = f"{safe_sample_name}_rank{rank+1}_cell{c_idx}"
                        self.visualize_residuals(c_raw, c_pre, os.path.join(output_dir, f"xai_residuals_{base_name}.png"))
                        self.visualize_heatmap_overlay(c_raw, c_pre, os.path.join(output_dir, f"xai_heatmap_{base_name}.png"))
                except Exception as e:
                    print(f"    [Warning] Failed XAI for {sample} rank {rank+1}: {e}")

        # --- Run Extended Analyses ---
        if generate_umap or run_extra_viz or run_quantitative:
            if not analysis_data['features']:
                print("No features found for extended analysis.")
                return
            
            all_features_pca = np.concatenate(analysis_data['features'], axis=0)
            analysis_df = pd.DataFrame({'sample': analysis_data['sample_name'], 'is_anomaly': analysis_data['is_anomaly']})
            
            samples = sorted(analysis_df['sample'].unique())
            palette = sns.color_palette("tab10", len(samples))
            color_map = dict(zip(samples, palette))

            if generate_umap:
                self.create_umap_visualization(all_features_pca, analysis_df, output_dir, color_map)
            if run_extra_viz:
                self.create_pca_visualization(all_features_pca, analysis_df, output_dir, color_map)
                self.create_tsne_visualization(all_features_pca, analysis_df, output_dir, color_map)
                self.create_phate_visualization(all_features_pca, analysis_df, output_dir, color_map)
            
            # Generate WT vs Mutant comparison plots if any visualization was created
            if generate_umap or run_extra_viz:
                self.create_wt_vs_mutant_visualizations(analysis_df, output_dir, color_map)

            if run_quantitative:
                self.calculate_distribution_distances(df_detailed, output_dir)
                self.perform_clustering_analysis(all_features_pca, analysis_df, output_dir, color_map)
        
        print(f"Folder mode processing complete. Results are in {output_dir}")

    # --- XAI Methods ---
    def visualize_residuals(self, raw_image, preprocessed_image, save_path):
        """
        Original (Raw), Reconstructed (from Preprocessed), and Difference Heatmap.
        """
        # Prepare input (1, 64, 64, 1)
        img_batch = np.expand_dims(preprocessed_image, axis=0)
        img_batch = np.expand_dims(img_batch, axis=-1)
        
        # Reconstruct
        reconstructed = self.autoencoder.predict(img_batch, verbose=0)
        reconstructed_img = reconstructed[0, :, :, 0]
        
        # Calculate residuals (Absolute difference between Preprocessed and Reconstructed)
        # Note: Usually residuals are calculated against the input to the AE (preprocessed).
        # The prompt asks to display 'Original' (raw) but reconstruct 'preprocessed'.
        # The difference should ideally be between what went in and what came out.
        diff = np.abs(preprocessed_image - reconstructed_img)
        
        # Plot
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.title("Original (Raw)")
        plt.imshow(raw_image, cmap='gray')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.title("Reconstructed")
        plt.imshow(reconstructed_img, cmap='gray')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.title("Difference Heatmap")
        plt.imshow(diff, cmap='inferno')
        plt.colorbar()
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def visualize_heatmap_overlay(self, raw_image, preprocessed_image, save_path):
        """
        Overlay reconstruction error heatmap on the raw image.
        """
        # Prepare input
        img_batch = np.expand_dims(preprocessed_image, axis=0)
        img_batch = np.expand_dims(img_batch, axis=-1)
        
        # Reconstruct
        reconstructed = self.autoencoder.predict(img_batch, verbose=0)
        reconstructed_img = reconstructed[0, :, :, 0]
        
        # Calculate residuals
        diff = np.abs(preprocessed_image - reconstructed_img)
        
        # Normalize diff for heatmap (0-1) for better visualization if needed, 
        # but keeping absolute values is more physically meaningful. 
        # However, for 'jet' colormap, it scales automatically if we use plt.imshow without vmin/vmax,
        # or we can fix it. Let's let matplotlib scale it.
        
        plt.figure(figsize=(6, 6))
        # 1. Show Raw Image in Gray
        plt.imshow(raw_image, cmap='gray')
        
        # 2. Overlay Heatmap
        # Use 'jet' colormap as requested, alpha=0.5
        plt.imshow(diff, cmap='jet', alpha=0.5)
        
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    # --- Standard Visualization Methods ---
    def plot_anomaly_rates(self, df, output_dir, thresholds, mode_name):
        plt.figure(figsize=(14, 7))
        names = [n[:20] for n in df['sample_name']]
        colors = ['blue' if is_wt else 'lightblue' for is_wt in df['is_wt']]
        plt.bar(range(len(names)), df['anomaly_rate'] * 100, color=colors, alpha=0.8)
        plt.axhline(thresholds['wt_rate'], color='blue', linestyle='--', label=f"WT Baseline ({thresholds['wt_rate']:.1f}%)")
        plt.axhline(thresholds['threshold'], color='red', linestyle='--', label=f"Hit Threshold ({thresholds['threshold']:.1f}%)")
        plt.xticks(range(len(names)), names, rotation=45, ha='right', fontsize=10)
        plt.ylabel('Anomaly Rate (%)'); plt.title(f'Anomaly Rates by Sample ({mode_name} Mode)'); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'plot_anomaly_rates_{mode_name.lower()}.png'), dpi=300); plt.close()

    def plot_violin(self, df_detailed, output_dir, thresholds, mode_name):
        print("  Generating Violin Plot...")
        samples = df_detailed['sample_name'].unique()
        wt_name = next((s for s in samples if s.upper() == 'WT'), None)

        # Plot 1: All samples together (original plot)
        plt.figure(figsize=(14, 8))
        order = [wt_name] + sorted([s for s in samples if s.upper() != 'WT']) if wt_name else sorted(samples)
        sns.violinplot(x='sample_name', y='anomaly_score', data=df_detailed, order=order, palette='Set2', inner='quartile')
        plt.axhline(thresholds['p99_score'], color='red', linestyle='--', linewidth=2, label=f'WT 99th Percentile ({thresholds["p99_score"]:.2f})')
        plt.legend(loc='upper right'); plt.title(f'Anomaly Score Distribution ({mode_name} Mode)'); plt.ylabel('Anomaly Score (Higher = More Abnormal)'); plt.xlabel('Sample')
        plt.xticks(rotation=45, ha='right'); plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'plot_violin_{mode_name.lower()}.png'), dpi=300); plt.close()

        # Plot 2: Individual WT vs Mutant plots
        if wt_name and mode_name.lower() == 'folder':
            mutants = [s for s in samples if s != wt_name]
            for mutant in mutants:
                plt.figure(figsize=(8, 6))
                sub_df = df_detailed[df_detailed['sample_name'].isin([wt_name, mutant])]
                sns.violinplot(x='sample_name', y='anomaly_score', data=sub_df, order=[wt_name, mutant], palette=['lightblue', 'salmon'])
                plt.axhline(thresholds['p99_score'], color='red', linestyle='--', linewidth=2, label=f'WT 99th Percentile ({thresholds["p99_score"]:.2f})')
                plt.legend(loc='upper right')
                plt.title(f'Anomaly Score: WT vs {mutant}')
                plt.ylabel('Anomaly Score')
                plt.xlabel('')
                plt.tight_layout()
                # To avoid issues with filenames, sanitize mutant names
                sanitized_mutant_name = "".join(c for c in mutant if c.isalnum() or c in ('-', '_')).rstrip()
                plt.savefig(os.path.join(output_dir, f'plot_violin_WT_vs_{sanitized_mutant_name}_{mode_name.lower()}.png'), dpi=300)
                plt.close()

    def generate_phenotype_mosaic(self, df_summary, df_detailed, output_dir, thresholds, mode, top_n_samples=5, top_n_cells=5):
        print("  Generating Phenotype Mosaic...")
        mutants = df_summary[~df_summary['is_wt']].sort_values('anomaly_rate', ascending=False).head(top_n_samples)
        targets = mutants['sample_name'].tolist()
        if 'WT' in df_summary.index and 'WT' not in targets:
            targets.insert(0, 'WT')
        if not targets: return
        fig, axes = plt.subplots(len(targets), top_n_cells, figsize=(top_n_cells * 2, len(targets) * 2.2))
        if len(targets) == 1: axes = np.array([axes])
        for r, sample_name in enumerate(targets):
            s_data = df_detailed[df_detailed['sample_name'] == sample_name]
            if s_data.empty: continue
            label_suffix, candidates = ("(Typical)", s_data.nsmallest(top_n_cells, 'anomaly_score')) if 'WT' in sample_name.upper() else ("(Anomaly)", s_data.nlargest(top_n_cells, 'anomaly_score'))
            axes[r, 0].text(-0.2, 0.5, f"{sample_name}\n{label_suffix}", transform=axes[r, 0].transAxes, va='center', ha='right', fontsize=11, fontweight='bold')
            if mode == 'file':
                # Unpack and take raw cell (index 0)
                cells_data = self.extract_quality_cells(df_summary.loc[sample_name, 'file_path'], enhance_contrast=False)
                raw_cells = [c[0] for c in cells_data]
            for c, (_, cand_row) in enumerate(candidates.iterrows()):
                ax = axes[r, c]
                if mode == 'folder':
                    # Unpack and take raw cell (index 0)
                    cells_data = self.extract_quality_cells(cand_row['file_path'], enhance_contrast=False)
                    raw_cells = [c[0] for c in cells_data]
                cell_idx = cand_row.get('local_idx', cand_row.get('cell_id'))
                if cell_idx < len(raw_cells):
                    img, score = raw_cells[cell_idx], cand_row['anomaly_score']
                    ax.imshow(img, cmap='gray', vmin=0, vmax=1)
                    ax.set_title(f"{score:.1f}", color='red' if score > thresholds['p99_score'] else 'black', fontsize=10, fontweight='bold')
                ax.axis('off')
        plt.suptitle("Phenotype Mosaic (Raw Images)"); plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(os.path.join(output_dir, f'plot_phenotype_mosaic_{mode.lower()}.png'), dpi=300); plt.close()

    # --- Extended Visualization & Analysis Methods ---
    def _plot_embedding(self, df, x_col, y_col, title, filename, color_map):
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=x_col, y=y_col, hue='sample', style='sample', size='is_anomaly', sizes=(10, 40), alpha=0.7, data=df, palette=color_map)
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[s], markersize=10) for s in color_map]
        plt.legend(handles, color_map.keys(), title='Strain', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title(title); plt.tight_layout()
        plt.savefig(filename, dpi=300); plt.close()

    def create_umap_visualization(self, features, df, output_dir, color_map):
        print("  Generating UMAP plot...")
        embedding = umap.UMAP(n_components=2, random_state=42).fit_transform(features)
        df['UMAP1'], df['UMAP2'] = embedding[:, 0], embedding[:, 1]
        self._plot_embedding(df, 'UMAP1', 'UMAP2', 'UMAP 2D Projection', os.path.join(output_dir, 'plot_umap.png'), color_map)

    def create_pca_visualization(self, features, df, output_dir, color_map):
        print("  Generating PCA plot...")
        df['PCA1'], df['PCA2'] = features[:, 0], features[:, 1]
        self._plot_embedding(df, 'PCA1', 'PCA2', 'PCA Projection (First 2 Components)', os.path.join(output_dir, 'plot_pca.png'), color_map)

    def create_tsne_visualization(self, features, df, output_dir, color_map):
        print("  Generating t-SNE plot...")
        embedding = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=300).fit_transform(features)
        df['tSNE1'], df['tSNE2'] = embedding[:, 0], embedding[:, 1]
        self._plot_embedding(df, 'tSNE1', 'tSNE2', 't-SNE 2D Projection', os.path.join(output_dir, 'plot_tsne.png'), color_map)

    def create_phate_visualization(self, features, df, output_dir, color_map):
        if phate is None:
            print("  Skipping PHATE plot: 'phate' library not installed.")
            return
        print("  Generating PHATE plot...")
        phate_op = phate.PHATE(random_state=42)
        embedding = phate_op.fit_transform(features)
        df['PHATE1'], df['PHATE2'] = embedding[:, 0], embedding[:, 1]
        self._plot_embedding(df, 'PHATE1', 'PHATE2', 'PHATE 2D Projection', os.path.join(output_dir, 'plot_phate.png'), color_map)

    def create_wt_vs_mutant_visualizations(self, df, output_dir, color_map):
        print("  Generating WT vs Mutant comparison plots...")
        wt_sample_name = next((s for s in df['sample'].unique() if s.upper() == 'WT'), None)
        
        if not wt_sample_name:
            print("  Skipping WT vs Mutant plots: WT sample not found.")
            return

        mutant_samples = [s for s in df['sample'].unique() if s != wt_sample_name]

        for mutant in mutant_samples:
            print(f"    - Comparing WT vs {mutant}")
            sub_df = df[df['sample'].isin([wt_sample_name, mutant])]
            comparison_color_map = {wt_sample_name: color_map.get(wt_sample_name, 'blue'), mutant: color_map.get(mutant, 'red')}
            
            sanitized_mutant_name = "".join(c for c in mutant if c.isalnum() or c in ('-', '_')).rstrip()

            # UMAP
            if 'UMAP1' in sub_df.columns:
                filename = os.path.join(output_dir, f'plot_umap_WT_vs_{sanitized_mutant_name}.png')
                self._plot_embedding(sub_df, 'UMAP1', 'UMAP2', f'UMAP: WT vs {mutant}', filename, comparison_color_map)
            
            # PCA
            if 'PCA1' in sub_df.columns:
                filename = os.path.join(output_dir, f'plot_pca_WT_vs_{sanitized_mutant_name}.png')
                self._plot_embedding(sub_df, 'PCA1', 'PCA2', f'PCA: WT vs {mutant}', filename, comparison_color_map)

            # t-SNE
            if 'tSNE1' in sub_df.columns:
                filename = os.path.join(output_dir, f'plot_tsne_WT_vs_{sanitized_mutant_name}.png')
                self._plot_embedding(sub_df, 'tSNE1', 'tSNE2', f't-SNE: WT vs {mutant}', filename, comparison_color_map)

            # PHATE
            if 'PHATE1' in sub_df.columns:
                if phate is not None:
                    filename = os.path.join(output_dir, f'plot_phate_WT_vs_{sanitized_mutant_name}.png')
                    self._plot_embedding(sub_df, 'PHATE1', 'PHATE2', f'PHATE: WT vs {mutant}', filename, comparison_color_map)

    def calculate_distribution_distances(self, df_detailed, output_dir):
        print("  Calculating distribution distances from WT...")
        wt_scores = df_detailed[df_detailed['sample_name'].str.upper() == 'WT']['anomaly_score']
        if wt_scores.empty:
            print("  Skipping distribution distances: WT sample not found.")
            return
        distances = {}
        for sample in df_detailed['sample_name'].unique():
            if sample.upper() == 'WT': continue
            sample_scores = df_detailed[df_detailed['sample_name'] == sample]['anomaly_score']
            distances[sample] = wasserstein_distance(wt_scores, sample_scores)
        df_dist = pd.DataFrame.from_dict(distances, orient='index', columns=['wasserstein_distance_from_WT']).sort_values(by='wasserstein_distance_from_WT', ascending=False)
        df_dist.to_csv(os.path.join(output_dir, 'quantitative_distribution_distances.csv'))
        print(f"  Saved distribution distances to {os.path.join(output_dir, 'quantitative_distribution_distances.csv')}")

    def perform_clustering_analysis(self, features, df, output_dir, color_map):
        if hdbscan is None:
            print("  Skipping Clustering: 'hdbscan' library not installed.")
            return
        print("  Performing HDBSCAN clustering...")
        clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=5)
        df['cluster'] = clusterer.fit_predict(features)
        
        # Composition analysis
        composition = df.groupby(['sample', 'cluster']).size().unstack(fill_value=0)
        composition_perc = composition.div(composition.sum(axis=1), axis=0) * 100
        composition_perc.to_csv(os.path.join(output_dir, 'quantitative_clustering_composition.csv'))
        
        # Plot composition heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(composition_perc, annot=True, fmt='.1f', cmap='viridis')
        plt.title('Cluster Composition (%) by Sample')
        plt.xlabel('Cluster ID (-1 = Outliers)')
        plt.ylabel('Sample')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'quantitative_clustering_heatmap.png'), dpi=300)
        plt.close()

        # Plot UMAP colored by cluster if UMAP data exists
        if 'UMAP1' in df.columns:
            plt.figure(figsize=(10, 8))
            unique_clusters = sorted(df['cluster'].unique())
            # Handle case with only outliers (-1)
            num_clusters = len(unique_clusters)
            palette = sns.color_palette("viridis", num_clusters - 1 if -1 in unique_clusters else num_clusters)
            
            cluster_colors = [c for c in unique_clusters if c != -1]
            color_map_cluster = {cluster: palette[i] for i, cluster in enumerate(cluster_colors)}
            if -1 in unique_clusters:
                color_map_cluster[-1] = (0.5, 0.5, 0.5)

            sns.scatterplot(x='UMAP1', y='UMAP2', hue='cluster', style='sample', data=df, palette=color_map_cluster, alpha=0.7)
            plt.title('UMAP Projection with HDBSCAN Clusters')
            plt.legend(title='Cluster ID', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'plot_umap_hdbscan_clusters.png'), dpi=300)
            plt.close()

        # WT cluster distribution bar plot
        wt_name = next((s for s in composition_perc.index if s.upper() == 'WT'), None)
        if wt_name:
            wt_composition = composition_perc.loc[wt_name]
            plt.figure(figsize=(8, 6))
            wt_composition.plot(kind='bar', color='skyblue')
            plt.title('Cluster Distribution for WT')
            plt.ylabel('Percentage (%)')
            plt.xlabel('Cluster ID')
            plt.xticks(rotation=0)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'quantitative_wt_cluster_distribution.png'), dpi=300)
            plt.close()
            
        print(f"  Saved clustering results to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Integrated Mutant Screening Pipeline.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--mode', required=True, choices=['file', 'folder', 'umap'], help="Analysis mode:\n'file':   Process each .tif file in INPUT_PATHS as a separate sample.\n'folder': Process each subfolder in INPUT_PATHS as a sample.\n'umap':   Same as 'folder' mode, but also generates UMAP plots.")
    parser.add_argument('--input_paths', required=True, nargs='+', help="One or more input files or directories.")
    parser.add_argument('--model_dir', required=True, help="Directory containing the trained models.")
    parser.add_argument('--output_dir', help="Directory to save results. If not given, a timestamped folder will be created.")
    parser.add_argument('--wt_path', help="Path to the WT file or folder for baseline calculation. If not provided, will look for 'WT' in inputs.")
    # --- New arguments for extended analysis ---
    parser.add_argument('--extra_viz', action='store_true', help="Generate extra visualizations (t-SNE, PCA, PHATE) in folder/umap mode.")
    parser.add_argument('--quantitative', action='store_true', help="Perform quantitative analysis (distribution distance, clustering) in folder/umap mode.")
    
    args = parser.parse_args()

    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"./screening_results/{timestamp}_{args.mode}_mode"
    
    try:
        pipeline = MutantScreeningPipeline(args.model_dir)
        if args.mode == 'file':
            pipeline.run_file_mode(args.input_paths, output_dir, args.wt_path)
        elif args.mode in ['folder', 'umap']:
            if len(args.input_paths) > 1:
                print("Warning: In 'folder' or 'umap' mode, only the first input path is used as the root directory.")
            root_path = args.input_paths[0]
            generate_umap = (args.mode == 'umap')
            pipeline.run_folder_mode(root_path, output_dir, generate_umap, args.extra_viz, args.quantitative, args.wt_path)
    except Exception as e:
        print(f"\n[CRITICAL ERROR] An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
