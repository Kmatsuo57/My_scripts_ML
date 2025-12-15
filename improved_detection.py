import numpy as np
import tifffile as tiff
import os
from glob import glob
from tensorflow.keras.models import load_model
from stardist.models import StarDist2D
from csbdeep.utils import normalize
from skimage.measure import regionprops, label as skimage_label
from skimage.segmentation import clear_border
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
        
    def get_files_from_folder(self, folder_path):
        """フォルダ内の全TIFファイルを取得してファイルベースの辞書を作成"""
        if not os.path.exists(folder_path):
            print(f"Folder not found: {folder_path}")
            return {}
        
        tif_files = sorted(glob(os.path.join(folder_path, '*.tif')))
        if not tif_files:
            print(f"No TIF files found in {folder_path}")
            return {}
        
        files_dict = {}
        for file_path in tif_files:
            # ファイル名（拡張子なし）を系列名として使用
            sample_name = os.path.splitext(os.path.basename(file_path))[0]
            files_dict[sample_name] = file_path
        
        print(f"Found {len(tif_files)} TIF files in {folder_path}")
        return files_dict
        
    def load_trained_models(self):
        """訓練済みモデルの読み込み"""
        print("Loading trained models...")
        
        # AutoencoderとEncoder
        self.autoencoder = load_model(os.path.join(self.model_dir, 'best_autoencoder.keras'))
        self.encoder = load_model(os.path.join(self.model_dir, 'encoder.keras'))
        
        # 前処理器
        with open(os.path.join(self.model_dir, 'scaler.pkl'), 'rb') as f:
            self.scaler = pickle.load(f)
        with open(os.path.join(self.model_dir, 'pca.pkl'), 'rb') as f:
            self.pca = pickle.load(f)
        
        # 異常検知器
        with open(os.path.join(self.model_dir, 'detector_conservative.pkl'), 'rb') as f:
            self.detector_conservative = pickle.load(f)
        with open(os.path.join(self.model_dir, 'detector_moderate.pkl'), 'rb') as f:
            self.detector_moderate = pickle.load(f)
        
        # StarDist
        self.stardist_model = StarDist2D.from_pretrained('2D_versatile_fluo')
        
        print("All models loaded successfully!")
    
    def extract_quality_cells(self, image_path):
        """品質管理付き細胞抽出（訓練時と同じ条件）"""
        try:
            image = tiff.imread(image_path)
            
            # チャンネル分離
            if image.ndim == 3 and image.shape[-1] >= 3:
                seg_channel = image[..., 2]  # セグメンテーション用
                green_channel = image[..., 1]  # 解析用
            else:
                seg_channel = image
                green_channel = image
            
            # StarDist セグメンテーション
            normalized_seg = normalize(seg_channel)
            labels, details = self.stardist_model.predict_instances(normalized_seg)
            
            # 品質フィルタリング（訓練時と同じ条件）
            height, width = labels.shape
            props = regionprops(labels)
            
            quality_cells = []
            cell_stats = []
            
            for prop in props:
                minr, minc, maxr, maxc = prop.bbox
                
                # 境界チェック
                if (minr < 10 or minc < 10 or maxr > (height - 10) or maxc > (width - 10)):
                    continue
                
                # サイズチェック
                if prop.area < 200 or prop.area > 8000:
                    continue
                
                # 形状チェック
                if prop.eccentricity > 0.95:
                    continue
                
                # 細胞画像抽出
                cell_image = green_channel[minr:maxr, minc:maxc]
                
                # 強度チェック
                cell_mean = np.mean(cell_image)
                cell_std = np.std(cell_image)
                
                if cell_mean < 0.5 or cell_std < 0.1:
                    continue
                
                # 訓練時と同じ前処理
                cell_image_eq = exposure.equalize_adapthist(cell_image, clip_limit=0.02)
                cell_image_resized = resize(cell_image_eq, (64, 64), anti_aliasing=True)
                
                quality_cells.append(cell_image_resized)
                
                cell_stats.append({
                    'area': prop.area,
                    'eccentricity': prop.eccentricity,
                    'solidity': prop.solidity,
                    'mean_intensity': cell_mean,
                    'std_intensity': cell_std
                })
            
            return quality_cells, cell_stats
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return [], []
    
    def compute_anomaly_scores(self, cell_images):
        """包括的異常スコア計算"""
        if len(cell_images) == 0:
            return {}
        
        X = np.expand_dims(np.array(cell_images), axis=-1).astype('float32')
        
        # 1. 再構成誤差
        reconstructed = self.autoencoder.predict(X, verbose=0)
        mse_errors = np.mean(np.square(X - reconstructed), axis=(1, 2, 3))
        mae_errors = np.mean(np.abs(X - reconstructed), axis=(1, 2, 3))
        
        # 2. エンコーダ特徴量ベース
        encoded_features = self.encoder.predict(X, verbose=0)
        encoded_flat = encoded_features.reshape(len(encoded_features), -1)
        
        # 前処理（訓練時と同じ）
        encoded_scaled = self.scaler.transform(encoded_flat)
        encoded_pca = self.pca.transform(encoded_scaled)
        
        # 異常検知
        conservative_predictions = self.detector_conservative.predict(encoded_pca)
        moderate_predictions = self.detector_moderate.predict(encoded_pca)
        
        conservative_scores = self.detector_conservative.decision_function(encoded_pca)
        moderate_scores = self.detector_moderate.decision_function(encoded_pca)
        
        return {
            'reconstruction_mse': mse_errors,
            'reconstruction_mae': mae_errors,
            'conservative_predictions': conservative_predictions,
            'moderate_predictions': moderate_predictions,
            'conservative_scores': -conservative_scores,  # 高いほど異常
            'moderate_scores': -moderate_scores,
            'conservative_anomaly_rate': np.sum(conservative_predictions == -1) / len(conservative_predictions),
            'moderate_anomaly_rate': np.sum(moderate_predictions == -1) / len(moderate_predictions)
        }
    
    def screen_mutant_samples(self, test_files_dict, output_dir):
        """変異株スクリーニング実行（ファイル単位、WT基準閾値）"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("=== Starting Mutant Screening with Improved Model (File-based) ===")
        
        results = {}
        detailed_results = []
        
        # まずWTファイルの異常率を計算して基準値を決定（必須）
        try:
            wt_thresholds = self.calculate_wt_baseline(test_files_dict)
        except Exception as e:
            print(f"[ERROR] Failed to compute WT baseline: {e}")
            print("Aborting screening because WT-based threshold is required.")
            return {}, []
        
        for sample_name, file_path in test_files_dict.items():
            print(f"\nProcessing {sample_name} ({os.path.basename(file_path)})...")
            
            if not os.path.exists(file_path):
                print(f"  File not found: {file_path}")
                # 空の結果を記録
                sample_result = {
                    'sample_name': sample_name,
                    'file_path': file_path,
                    'total_cells': 0,
                    'conservative_anomaly_rate': 0.0,
                    'moderate_anomaly_rate': 0.0,
                    'mean_mse': 0.0,
                    'std_mse': 0.0,
                    'mean_mae': 0.0,
                    'std_mae': 0.0,
                    'is_wt': sample_name.upper() == 'WT'
                }
                results[sample_name] = sample_result
                continue
            
            # 単一ファイルの処理
            cells, stats = self.extract_quality_cells(file_path)
            
            print(f"  Extracted {len(cells)} cells from {os.path.basename(file_path)}")
            
            if len(cells) == 0:
                print(f"  No quality cells extracted from {sample_name}")
                # 空の結果も記録
                sample_result = {
                    'sample_name': sample_name,
                    'file_path': file_path,
                    'total_cells': 0,
                    'conservative_anomaly_rate': 0.0,
                    'moderate_anomaly_rate': 0.0,
                    'mean_mse': 0.0,
                    'std_mse': 0.0,
                    'mean_mae': 0.0,
                    'std_mae': 0.0,
                    'is_wt': sample_name.upper() == 'WT'
                }
                results[sample_name] = sample_result
                continue
            
            # 異常スコア計算
            anomaly_scores = self.compute_anomaly_scores(cells)
            
            # 結果サマリー
            sample_result = {
                'sample_name': sample_name,
                'file_path': file_path,
                'total_cells': len(cells),
                'conservative_anomaly_rate': anomaly_scores['conservative_anomaly_rate'],
                'moderate_anomaly_rate': anomaly_scores['moderate_anomaly_rate'],
                'mean_mse': np.mean(anomaly_scores['reconstruction_mse']),
                'std_mse': np.std(anomaly_scores['reconstruction_mse']),
                'mean_mae': np.mean(anomaly_scores['reconstruction_mae']),
                'std_mae': np.std(anomaly_scores['reconstruction_mae']),
                'is_wt': sample_name.upper() == 'WT'
            }
            
            results[sample_name] = sample_result
            
            # 詳細結果（細胞レベル）
            for i, (mse, mae, cons_pred, mod_pred, cons_score, mod_score) in enumerate(zip(
                anomaly_scores['reconstruction_mse'],
                anomaly_scores['reconstruction_mae'],
                anomaly_scores['conservative_predictions'],
                anomaly_scores['moderate_predictions'],
                anomaly_scores['conservative_scores'],
                anomaly_scores['moderate_scores']
            )):
                detailed_results.append({
                    'sample_name': sample_name,
                    'file_name': os.path.basename(file_path),
                    'cell_id': i,
                    'mse': mse,
                    'mae': mae,
                    'conservative_anomaly': cons_pred == -1,
                    'moderate_anomaly': mod_pred == -1,
                    'conservative_score': cons_score,
                    'moderate_score': mod_score
                })
            
            # 進捗表示
            print(f"    Conservative anomaly rate: {sample_result['conservative_anomaly_rate']*100:.2f}%")
            print(f"    Moderate anomaly rate: {sample_result['moderate_anomaly_rate']*100:.2f}%")
            print(f"    Mean MSE: {sample_result['mean_mse']:.6f}")
        
        # 結果の保存と可視化（空の結果でも実行）
        if results:  # 結果がある場合のみ
            self.save_and_visualize_results(results, detailed_results, output_dir, wt_thresholds)
        else:
            print("No results to save or visualize.")
        
        return results, detailed_results
    
    def screen_mutant_samples_by_folder(self, test_folders_dict, output_dir):
        """変異株スクリーニング実行（フォルダ単位、WT基準閾値）"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("=== Starting Mutant Screening with Improved Model (Folder-based) ===")
        
        results = {}
        detailed_results = []
        
        # まずWTファイルの異常率を計算（固定パスから）
        fixed_wt_path = "/Users/matsuokoujirou/Documents/Data/Screening/Pools/normal/WT.tif"
        wt_files_dict = {'WT': fixed_wt_path} if os.path.exists(fixed_wt_path) else {}
        
        try:
            wt_thresholds = self.calculate_wt_baseline(wt_files_dict)
        except Exception as e:
            print(f"[ERROR] Failed to compute WT baseline: {e}")
            print("Aborting screening because WT-based threshold is required.")
            return {}, []
        
        for folder_name, folder_path in test_folders_dict.items():
            print(f"\nProcessing folder: {folder_name} ({folder_path})...")
            
            if not os.path.exists(folder_path):
                print(f"  Folder not found: {folder_path}")
                continue
            
            # フォルダ内の全TIFファイルを取得
            tif_files = sorted(glob(os.path.join(folder_path, '*.tif')))
            if not tif_files:
                print(f"  No .tif files found in {folder_path}")
                continue
            
            print(f"  Found {len(tif_files)} TIF files in folder")
            
            # フォルダ内の全ファイルから細胞を抽出
            all_cells = []
            file_summary = []
            
            for file_path in tif_files:
                filename = os.path.basename(file_path)
                cells, stats = self.extract_quality_cells(file_path)
                
                num_cells = len(cells)
                all_cells.extend(cells)
                file_summary.append({
                    'filename': filename,
                    'cells_extracted': num_cells,
                    'start_idx': len(all_cells) - num_cells,
                    'end_idx': len(all_cells)
                })
                
                print(f"    {filename}: {num_cells} cells")
            
            print(f"  Total cells extracted from {folder_name}: {len(all_cells)}")
            
            if len(all_cells) == 0:
                print(f"  No quality cells extracted from {folder_name}")
                # 空の結果も記録
                sample_result = {
                    'sample_name': folder_name,
                    'folder_path': folder_path,
                    'total_cells': 0,
                    'files_processed': len(tif_files),
                    'conservative_anomaly_rate': 0.0,
                    'moderate_anomaly_rate': 0.0,
                    'mean_mse': 0.0,
                    'std_mse': 0.0,
                    'mean_mae': 0.0,
                    'std_mae': 0.0,
                    'is_wt': folder_name.upper() == 'WT'
                }
                results[folder_name] = sample_result
                continue
            
            # 異常スコア計算（フォルダ内の全細胞をまとめて）
            anomaly_scores = self.compute_anomaly_scores(all_cells)
            
            # 結果サマリー
            sample_result = {
                'sample_name': folder_name,
                'folder_path': folder_path,
                'total_cells': len(all_cells),
                'files_processed': len(tif_files),
                'conservative_anomaly_rate': anomaly_scores['conservative_anomaly_rate'],
                'moderate_anomaly_rate': anomaly_scores['moderate_anomaly_rate'],
                'mean_mse': np.mean(anomaly_scores['reconstruction_mse']),
                'std_mse': np.std(anomaly_scores['reconstruction_mse']),
                'mean_mae': np.mean(anomaly_scores['reconstruction_mae']),
                'std_mae': np.std(anomaly_scores['reconstruction_mae']),
                'is_wt': folder_name.upper() == 'WT'
            }
            
            results[folder_name] = sample_result
            
            # 詳細結果（細胞レベル）
            # 各ファイルから抽出された細胞を追跡
            for file_info in file_summary:
                filename = file_info['filename']
                start_idx = file_info['start_idx']
                end_idx = file_info['end_idx']
                
                # このファイルの細胞分の結果を記録
                for cell_idx in range(start_idx, end_idx):
                    if cell_idx < len(anomaly_scores['reconstruction_mse']):
                        detailed_results.append({
                            'sample_name': folder_name,
                            'file_name': filename,
                            'cell_id': cell_idx,
                            'mse': anomaly_scores['reconstruction_mse'][cell_idx],
                            'mae': anomaly_scores['reconstruction_mae'][cell_idx],
                            'conservative_anomaly': anomaly_scores['conservative_predictions'][cell_idx] == -1,
                            'moderate_anomaly': anomaly_scores['moderate_predictions'][cell_idx] == -1,
                            'conservative_score': anomaly_scores['conservative_scores'][cell_idx],
                            'moderate_score': anomaly_scores['moderate_scores'][cell_idx]
                        })
            
            # 進捗表示
            print(f"    Conservative anomaly rate: {sample_result['conservative_anomaly_rate']*100:.2f}%")
            print(f"    Moderate anomaly rate: {sample_result['moderate_anomaly_rate']*100:.2f}%")
            print(f"    Mean MSE: {sample_result['mean_mse']:.6f}")
        
        # 結果の保存と可視化（空の結果でも実行）
        if results:  # 結果がある場合のみ
            self.save_and_visualize_results(results, detailed_results, output_dir, wt_thresholds)
        else:
            print("No results to save or visualize.")
        
        return results, detailed_results
    
    def calculate_wt_baseline(self, test_files_dict):
        """WTファイルの異常率を計算して基準値を決定"""
        print("\n=== Calculating WT Baseline ===")
        
        wt_file_path = None
        for sample_name, file_path in test_files_dict.items():
            if sample_name.upper() == 'WT':
                wt_file_path = file_path
                break
        
        if not wt_file_path or not os.path.exists(wt_file_path):
            # フォルダにWTが無い場合は固定パスのWTを使用
            fixed_wt_path = "/Users/matsuokoujirou/Documents/Data/Screening/Pools/normal/WT.tif"
            if os.path.exists(fixed_wt_path):
                wt_file_path = fixed_wt_path
                print(f"  WT file not found in test set. Using fixed WT: {fixed_wt_path}")
            else:
                raise FileNotFoundError(
                    "WT file not found in test set and fixed WT path does not exist: "
                    "/Users/matsuokoujirou/Documents/Data/Screening/Pools/normal/WT.tif"
                )
        
        print(f"  Processing WT file: {os.path.basename(wt_file_path)}")
        
        # WTファイルの処理
        cells, stats = self.extract_quality_cells(wt_file_path)
        
        if len(cells) == 0:
            raise ValueError("No quality cells extracted from WT file; cannot compute WT baseline.")
        
        print(f"  Extracted {len(cells)} cells from WT file")
        
        # WTの異常スコア計算
        anomaly_scores = self.compute_anomaly_scores(cells)
        
        wt_conservative_rate = anomaly_scores['conservative_anomaly_rate'] * 100
        wt_moderate_rate = anomaly_scores['moderate_anomaly_rate'] * 100
        
        # 閾値計算（WT異常率 + 4.2%）
        threshold_conservative = wt_conservative_rate + 4.2
        threshold_moderate = wt_moderate_rate + 4.2
        
        print(f"  WT Conservative rate: {wt_conservative_rate:.2f}%")
        print(f"  WT Moderate rate: {wt_moderate_rate:.2f}%")
        print(f"  Threshold Conservative: {threshold_conservative:.2f}%")
        print(f"  Threshold Moderate: {threshold_moderate:.2f}%")
        
        return {
            'wt_conservative_rate': wt_conservative_rate,
            'wt_moderate_rate': wt_moderate_rate,
            'threshold_conservative': threshold_conservative,
            'threshold_moderate': threshold_moderate
        }
    
    def save_and_visualize_results(self, results, detailed_results, output_dir, wt_thresholds):
        """結果の保存と可視化（WT基準閾値）"""
        
        # サマリー結果のDataFrame化
        results_df = pd.DataFrame.from_dict(results, orient='index')
        results_df.to_csv(os.path.join(output_dir, 'screening_summary.csv'))
        
        # 詳細結果の保存
        detailed_df = pd.DataFrame(detailed_results)
        detailed_df.to_csv(os.path.join(output_dir, 'detailed_cell_results.csv'), index=False)
        
        # 可視化
        self.create_screening_visualizations(results_df, detailed_df, output_dir, wt_thresholds)
        
        # レポート生成
        self.generate_screening_report(results_df, output_dir, wt_thresholds)
    
    def create_screening_visualizations(self, results_df, detailed_df, output_dir, wt_thresholds):
        """スクリーニング結果の可視化（WT基準閾値）"""
        
        # 1. 異常率比較
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # ファイル名を短縮表示用に調整
        display_names = [name[:15] + '...' if len(name) > 15 else name for name in results_df['sample_name']]
        
        conservative_rates = results_df['conservative_anomaly_rate'] * 100
        moderate_rates = results_df['moderate_anomaly_rate'] * 100
        
        # Conservative
        bars1 = ax1.bar(range(len(display_names)), conservative_rates, color='lightblue', alpha=0.8)
        # WT基準の閾値ライン
        # ax1.axhline(y=wt_thresholds['wt_conservative_rate'], color='blue', linestyle='--', alpha=0.7, 
        #            label=f'WT Baseline ({wt_thresholds["wt_conservative_rate"]:.1f}%)')
        # ax1.axhline(y=wt_thresholds['threshold_conservative'], color='red', linestyle='--', alpha=0.7, 
        #            label=f'Anomaly Threshold ({wt_thresholds["threshold_conservative"]:.1f}%)')
        ax1.set_title('Conservative Model - Anomaly Rates (WT-based Threshold)')
        ax1.set_ylabel('Anomaly Rate (%)')
        ax1.set_xticks(range(len(display_names)))
        ax1.set_xticklabels(display_names, rotation=45, ha='right', fontweight='bold', fontsize=18)
        ax1.title.set_fontsize(18)

        # # 値の表示（ファイル数が多い場合は一部のみ）
        # if len(display_names) <= 20:
        #     for bar, rate in zip(bars1, conservative_rates):
        #         ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
        #                 f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=8)
        
        # Moderate
        bars2 = ax2.bar(range(len(display_names)), moderate_rates, color='royalblue', alpha=0.8)
        # WT基準の閾値ライン
        # ax2.axhline(y=wt_thresholds['wt_moderate_rate'], color='blue', linestyle='--', alpha=0.7, 
        #            label=f'WT Baseline ({wt_thresholds["wt_moderate_rate"]:.1f}%)')
        # ax2.axhline(y=wt_thresholds['threshold_moderate'], color='red', linestyle='--', alpha=0.7, 
        #            label=f'Anomaly Threshold ({wt_thresholds["threshold_moderate"]:.1f}%)')
        ax2.set_title('Moderate Model - Anomaly Rates (WT-based Threshold)')
        ax2.set_ylabel('Anomaly Rate (%)')
        ax2.set_xticks(range(len(display_names)))
        ax2.set_xticklabels(display_names, rotation=45, ha='right', fontweight='bold', fontsize=18)
        ax2.title.set_fontsize(18)
        
        # if len(display_names) <= 20:
        #     for bar, rate in zip(bars2, moderate_rates):
        #         ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
        #                     f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'anomaly_rates_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. 再構成誤差分布
        # フォルダベースかどうかを判定
        is_folder_based = 'folder_path' in results_df.columns or 'files_processed' in results_df.columns
        
        if is_folder_based:
            # フォルダベース: WTと重ねて表示
            # WTデータを取得
            wt_data = detailed_df[detailed_df['sample_name'].str.upper() == 'WT']
            wt_mse = wt_data['mse'].values if len(wt_data) > 0 else None
            
            # WT以外のフォルダを取得
            non_wt_samples = [name for name in results_df['sample_name'].unique() if name.upper() != 'WT']
            n_samples = len(non_wt_samples)
            
            if n_samples > 0:
                fig, axes = plt.subplots(2, (n_samples + 1) // 2, figsize=(16, 10))
                axes = axes.flatten() if n_samples > 1 else [axes]
                
                for i, sample_name in enumerate(non_wt_samples):
                    if i >= len(axes):
                        break
                        
                    sample_data = detailed_df[detailed_df['sample_name'] == sample_name]
                    
                    if len(sample_data) > 0:
                        # サンプルのMSE分布
                        axes[i].hist(sample_data['mse'], bins=30, alpha=0.6, density=True, 
                                   color='lightcoral', label=sample_name)
                        
                        # WTのMSE分布を重ねて表示
                        if wt_mse is not None and len(wt_mse) > 0:
                            axes[i].hist(wt_mse, bins=30, alpha=0.6, density=True, 
                                       color='lightblue', label='WT')
                        
                        axes[i].set_title(f'{sample_name[:20]}', fontweight='bold', fontsize=18)
                        axes[i].set_xlabel('MSE', fontsize=18)
                        axes[i].set_ylabel('Density', fontsize=18)
                        
                        # 平均線
                        mean_mse = sample_data['mse'].mean()
                        axes[i].axvline(mean_mse, color='red', linestyle='--', 
                                      label=f'{sample_name} Mean: {mean_mse:.4f}', linewidth=2)
                        
                        if wt_mse is not None and len(wt_mse) > 0:
                            wt_mean = np.mean(wt_mse)
                            axes[i].axvline(wt_mean, color='blue', linestyle='--', 
                                          label=f'WT Mean: {wt_mean:.4f}', linewidth=2)
                        
                        # axes[i].legend(fontsize=8)
                
                # 空のサブプロットを隠す
                for i in range(len(non_wt_samples), len(axes)):
                    axes[i].set_visible(False)
                
                plt.suptitle('MSE Distributions - Folders vs WT (Overlaid)', fontsize=14)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'mse_distributions.png'), dpi=300, bbox_inches='tight')
                plt.show()
        elif len(results_df) <= 12:  # 全ファイル表示（ファイルベース）
            sample_names = results_df['sample_name'].unique()
            n_samples = len(sample_names)
            
            fig, axes = plt.subplots(2, (n_samples + 1) // 2, figsize=(16, 10))
            axes = axes.flatten() if n_samples > 1 else [axes]
            
            for i, sample_name in enumerate(sample_names):
                if i >= len(axes):
                    break
                    
                sample_data = detailed_df[detailed_df['sample_name'] == sample_name]
                
                if len(sample_data) > 0:
                    # WTファイルを特別に表示
                    is_wt = sample_name.upper() == 'WT'
                    color = 'lightblue' if is_wt else 'lightcoral'
                    title_color = 'blue' if is_wt else 'black'
                    
                    axes[i].hist(sample_data['mse'], bins=30, alpha=0.7, density=True, color=color)
                    axes[i].set_title(f'{sample_name[:20]}...\n(n={len(sample_data)})', 
                                    color=title_color, fontweight='bold' if is_wt else 'normal')
                    axes[i].set_xlabel('MSE')
                    axes[i].set_ylabel('Density')
                    
                    # 平均線
                    mean_mse = sample_data['mse'].mean()
                    line_color = 'blue' if is_wt else 'red'
                    axes[i].axvline(mean_mse, color=line_color, linestyle='--', 
                                  label=f'Mean: {mean_mse:.4f}', linewidth=2)
                    axes[i].legend()
            
            # 空のサブプロットを隠す
            for i in range(len(sample_names), len(axes)):
                axes[i].set_visible(False)
            
            plt.suptitle('MSE Distributions - All Files (Blue=WT)', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'mse_distributions.png'), dpi=300, bbox_inches='tight')
            plt.show()
        else:  # 上位ファイル + WT表示（ファイルベース）
            # 異常率が高い上位11ファイル + WT
            top_anomaly = results_df.nlargest(11, 'conservative_anomaly_rate')
            
            # WTファイルを追加（まだ含まれていない場合）
            wt_files = results_df[results_df.get('is_wt', False)]
            if not wt_files.empty and not any(row['sample_name'] == 'WT' for _, row in top_anomaly.iterrows()):
                top_anomaly = pd.concat([top_anomaly, wt_files])
            
            fig, axes = plt.subplots(3, 4, figsize=(16, 12))
            axes = axes.flatten()
            
            for i, (_, row) in enumerate(top_anomaly.iterrows()):
                sample_name = row['sample_name']
                sample_data = detailed_df[detailed_df['sample_name'] == sample_name]
                
                if len(sample_data) > 0:
                    # WTファイルは特別な色で表示
                    color = 'lightblue' if row.get('is_wt', False) else 'lightcoral'
                    title_color = 'blue' if row.get('is_wt', False) else 'black'
                    
                    axes[i].hist(sample_data['mse'], bins=20, alpha=0.7, density=True, color=color)
                    axes[i].set_title(f'{sample_name[:15]}...\n(n={len(sample_data)})', 
                                    fontsize=10, color=title_color, fontweight='bold' if row.get('is_wt', False) else 'normal')
                    axes[i].set_xlabel('MSE', fontsize=9)
                    axes[i].set_ylabel('Density', fontsize=9)
                    
                    mean_mse = sample_data['mse'].mean()
                    line_color = 'blue' if row.get('is_wt', False) else 'red'
                    axes[i].axvline(mean_mse, color=line_color, linestyle='--', alpha=0.7, linewidth=2)
            
            # 空のサブプロットを隠す
            for i in range(len(top_anomaly), len(axes)):
                axes[i].set_visible(False)
            
            plt.suptitle('MSE Distributions - Top Anomaly Files + WT (Blue=WT)', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'mse_distributions_top.png'), dpi=300, bbox_inches='tight')
            plt.show()
        
        # 3. 相関行列
        if len(results_df) > 2:
            plt.figure(figsize=(10, 8))
            
            correlation_data = results_df[['conservative_anomaly_rate', 'moderate_anomaly_rate', 'mean_mse', 'mean_mae']]
            correlation_matrix = correlation_data.corr()
            
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, cbar_kws={'label': 'Correlation Coefficient'})
            plt.title('Correlation Matrix of Anomaly Metrics (File-based)')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')
            plt.show()
    
        # 4. ファイル別細胞数と異常率の散布図
        plt.figure(figsize=(12, 8))
        
        scatter = plt.scatter(results_df['total_cells'], 
                             results_df['conservative_anomaly_rate'] * 100,
                             c=results_df['mean_mse'], 
                             cmap='viridis', 
                             alpha=0.7, 
                             s=100)
        
        plt.xlabel('Total Cells per File')
        plt.ylabel('Conservative Anomaly Rate (%)')
        plt.title('Cell Count vs Anomaly Rate (WT-based Threshold)')
        plt.colorbar(scatter, label='Mean MSE')
        
        # WT基準の閾値ライン
        plt.axhline(y=wt_thresholds['wt_conservative_rate'], color='blue', linestyle='--', alpha=0.7, 
                   label=f'WT Baseline ({wt_thresholds["wt_conservative_rate"]:.1f}%)')
        plt.axhline(y=wt_thresholds['threshold_conservative'], color='red', linestyle='--', alpha=0.7, 
                   label=f'Anomaly Threshold ({wt_thresholds["threshold_conservative"]:.1f}%)')
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cells_vs_anomaly_scatter.png'), dpi=300, bbox_inches='tight')
        plt.show()

        # 5. Conservative異常率ヒートマップ（横10列）
        names = results_df['sample_name'].tolist()
        rates_percent = (results_df['conservative_anomaly_rate'] * 100).tolist()
        is_wt_list = results_df.get('is_wt', pd.Series([False] * len(results_df), index=results_df.index)).tolist()

        cols = 10
        n = len(names)
        rows = (n + cols - 1) // cols

        heat = np.full((rows, cols), np.nan, dtype=float)
        name_grid = [[None for _ in range(cols)] for _ in range(rows)]
        wt_pos = None

        for idx, (name, rate, is_wt) in enumerate(zip(names, rates_percent, is_wt_list)):
            r = idx // cols
            c = idx % cols
            heat[r, c] = rate
            name_grid[r][c] = name
            if is_wt and wt_pos is None:
                wt_pos = (r, c)

        # 表示範囲を0-15%に固定
        vmax = 15.0

        fig, ax = plt.subplots(figsize=(cols * 1.0, rows * 1.1 + 0.5))
        im = ax.imshow(heat, cmap='Reds', vmin=0.0, vmax=vmax)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Conservative Anomaly Rate (%)')

        for r in range(rows):
            for c in range(cols):
                if not np.isnan(heat[r, c]):
                    label_name = name_grid[r][c]
                    rate = heat[r, c]
                    ax.text(c, r, f"{label_name}\n{rate:.1f}%", ha='center', va='center', fontsize=7)

        # WTを青枠で強調表示
        if wt_pos is not None:
            r, c = wt_pos
            rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor='blue', linewidth=2)
            ax.add_patch(rect)

        ax.set_title('Conservative Anomaly Rate Heatmap (WT-based Threshold)')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'conservative_rate_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_screening_report(self, results_df, output_dir, wt_thresholds):
        """スクリーニングレポート生成（WT基準閾値）"""
        
        with open(os.path.join(output_dir, 'mutant_screening_report.txt'), 'w') as f:
            f.write("=== MUTANT SCREENING REPORT (WT-BASED THRESHOLD) ===\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("WT BASELINE AND THRESHOLDS:\n")
            f.write(f"- WT Conservative rate: {wt_thresholds['wt_conservative_rate']:.2f}%\n")
            f.write(f"- WT Moderate rate: {wt_thresholds['wt_moderate_rate']:.2f}%\n")
            f.write(f"- Anomaly threshold Conservative: {wt_thresholds['threshold_conservative']:.2f}% (WT + 4.2%)\n")
            f.write(f"- Anomaly threshold Moderate: {wt_thresholds['threshold_moderate']:.2f}% (WT + 4.2%)\n\n")
            
            # ファイルベースかフォルダベースかを判定
            is_folder_based = 'folder_path' in results_df.columns or 'files_processed' in results_df.columns
            
            if is_folder_based:
                f.write("SCREENING RESULTS (FOLDER-BASED):\n")
                f.write("-" * 120 + "\n")
                f.write(f"{'Folder':<30} {'Cells':<8} {'Files':<8} {'Conservative':<12} {'Moderate':<12} {'Mean MSE':<12} {'Folder Path':<40}\n")
                f.write("-" * 120 + "\n")
                
                for _, row in results_df.iterrows():
                    folder_name = row['sample_name']
                    files_count = row.get('files_processed', 0)
                    folder_path = row.get('folder_path', 'N/A')
                    f.write(f"{folder_name:<30} {row['total_cells']:<8} {files_count:<8} "
                           f"{row['conservative_anomaly_rate']*100:>8.1f}% "
                           f"{row['moderate_anomaly_rate']*100:>10.1f}% "
                           f"{row['mean_mse']:>10.6f} "
                           f"{folder_path:<40}\n")
            else:
                f.write("SCREENING RESULTS (FILE-BASED):\n")
                f.write("-" * 120 + "\n")
                f.write(f"{'File':<30} {'Cells':<8} {'Conservative':<12} {'Moderate':<12} {'Mean MSE':<12} {'File Path':<40}\n")
                f.write("-" * 120 + "\n")
                
                for _, row in results_df.iterrows():
                    file_name = os.path.basename(row['file_path']) if 'file_path' in row else row['sample_name']
                    f.write(f"{file_name:<30} {row['total_cells']:<8} "
                       f"{row['conservative_anomaly_rate']*100:>8.1f}% "
                       f"{row['moderate_anomaly_rate']*100:>10.1f}% "
                           f"{row['mean_mse']:>10.6f} "
                           f"{row.get('file_path', 'N/A'):<40}\n")
            
            f.write("\n")
            
            # 統計サマリー
            f.write("STATISTICAL SUMMARY:\n")
            if is_folder_based:
                f.write(f"- Total folders processed: {len(results_df)}\n")
                total_files = results_df.get('files_processed', pd.Series([0] * len(results_df))).sum()
                f.write(f"- Total files processed: {total_files}\n")
                f.write(f"- Total cells analyzed: {results_df['total_cells'].sum()}\n")
                f.write(f"- Average cells per folder: {results_df['total_cells'].mean():.1f}\n")
            else:
                f.write(f"- Total files processed: {len(results_df)}\n")
                f.write(f"- Total cells analyzed: {results_df['total_cells'].sum()}\n")
                f.write(f"- Average cells per file: {results_df['total_cells'].mean():.1f}\n")
            f.write(f"- Average conservative anomaly rate: {results_df['conservative_anomaly_rate'].mean()*100:.2f}%\n")
            f.write(f"- Average moderate anomaly rate: {results_df['moderate_anomaly_rate'].mean()*100:.2f}%\n\n")
            
            # 異常候補の特定（WT基準閾値）
            f.write("ANOMALY ANALYSIS (WT-BASED THRESHOLDS):\n")
            
            # Conservative modelで閾値以上
            threshold_conservative_decimal = wt_thresholds['threshold_conservative'] / 100
            high_conservative = results_df[results_df['conservative_anomaly_rate'] > threshold_conservative_decimal]
            if not high_conservative.empty:
                unit = "folders" if is_folder_based else "files"
                f.write(f"\nHIGH ANOMALY CANDIDATES (Conservative >{wt_thresholds['threshold_conservative']:.1f}%): {len(high_conservative)} {unit}\n")
                for _, row in high_conservative.iterrows():
                    name = row['sample_name']
                    f.write(f"- {name}: {row['conservative_anomaly_rate']*100:.1f}% "
                           f"({row['total_cells']} cells)\n")
            
            # Moderate modelで閾値以上
            threshold_moderate_decimal = wt_thresholds['threshold_moderate'] / 100
            high_moderate = results_df[results_df['moderate_anomaly_rate'] > threshold_moderate_decimal]
            if not high_moderate.empty:
                unit = "folders" if is_folder_based else "files"
                f.write(f"\nHIGH ANOMALY CANDIDATES (Moderate >{wt_thresholds['threshold_moderate']:.1f}%): {len(high_moderate)} {unit}\n")
                for _, row in high_moderate.iterrows():
                    name = row['sample_name']
                    f.write(f"- {name}: {row['moderate_anomaly_rate']*100:.1f}% "
                           f"({row['total_cells']} cells)\n")
            
            # 正常レベル（WT異常率以下）
            wt_conservative_decimal = wt_thresholds['wt_conservative_rate'] / 100
            normal_conservative = results_df[results_df['conservative_anomaly_rate'] <= wt_conservative_decimal]
            if not normal_conservative.empty:
                unit = "folders" if is_folder_based else "files"
                f.write(f"\nNORMAL-LEVEL {unit.upper()} (Conservative ≤{wt_thresholds['wt_conservative_rate']:.1f}%): {len(normal_conservative)} {unit}\n")
                for _, row in normal_conservative.iterrows():
                    name = row['sample_name']
                    f.write(f"- {name}: {row['conservative_anomaly_rate']*100:.1f}% "
                           f"({row['total_cells']} cells)\n")
            
            # 細胞数による分類
            f.write("\nCELL COUNT ANALYSIS:\n")
            unit = "folders" if is_folder_based else "files"
            low_cell_items = results_df[results_df['total_cells'] < 50]
            if not low_cell_items.empty:
                f.write(f"- Low cell count {unit} (<50 cells): {len(low_cell_items)} {unit}\n")
            
            high_cell_items = results_df[results_df['total_cells'] > 200]
            if not high_cell_items.empty:
                f.write(f"- High cell count {unit} (>200 cells): {len(high_cell_items)} {unit}\n")
            
            f.write("\n\nRECOMMENDATIONS:\n")
            unit = "folders" if is_folder_based else "files"
            f.write(f"1. Focus on {unit} with Conservative >{wt_thresholds['threshold_conservative']:.1f}% for detailed analysis\n")
            f.write(f"2. {unit.capitalize()} with Conservative ≤{wt_thresholds['wt_conservative_rate']:.1f}% are likely normal phenotype\n")
            f.write("3. Consider cell count when interpreting results (low counts may be unreliable)\n")
            f.write(f"4. {unit.capitalize()} with high anomaly rates should be prioritized for validation\n")
            f.write("5. Validate results with independent experimental methods\n")
            f.write("6. WT-based thresholds provide experiment-specific normalization\n")


def main():
    """メイン実行関数（ファイルベース）"""
    # 設定
    model_dir = "/Users/matsuokoujirou/Documents/Data/Screening/Models/20250613_1524_improved"
    
    # スクリーナー初期化
    screener = ProductionMutantScreening(model_dir)
    
    # # 方法1: フォルダ内の全TIFファイルを自動検出
    folder_path = "/Users/matsuokoujirou/Documents/Data/Screening/Candidates/251127"
    test_files = screener.get_files_from_folder(folder_path)
    
    # 方法2: 個別ファイルを手動指定する場合（コメントアウトを解除して使用）
    # test_files = {
    #     "sample_001": "/path/to/your/file1.tif",
    #     "sample_002": "/path/to/your/file2.tif",
    #     # 必要に応じて追加
    # }
    
    方法3: 複数フォルダからファイルを取得する場合（1ファイル1系列）
    # folders = [
    #     "/Users/matsuokoujirou/Documents/Data/Screening/Abnormal_data/epyc1",
    #     "/Users/matsuokoujirou/Documents/Data/Screening/Abnormal_data/mith1",
    #     "/Users/matsuokoujirou/Documents/Data/Screening/Abnormal_data/rbmp1",
    #     "/Users/matsuokoujirou/Documents/Data/Screening/Abnormal_data/rbmp2",
    #     "/Users/matsuokoujirou/Documents/Data/Screening/Abnormal_data/RG2",
    #     "/Users/matsuokoujirou/Documents/Data/Screening/Abnormal_data/saga1",
    #     "/Users/matsuokoujirou/Documents/Data/Screening/Abnormal_data/saga2",
    # ]
    # test_files = {}
    # for folder in folders:
    #     folder_files = screener.get_files_from_folder(folder)
    #     test_files.update(folder_files)
    
    # 方法4: フォルダを1系列として処理（フォルダ内の全ファイルをまとめて）
    # test_folders = {
    #     "WT": "/Users/matsuokoujirou/Documents/Data/Screening/Pools/normal/RG2",
    #     "epyc1": "/Users/matsuokoujirou/Documents/Data/Screening/Abnormal_data/epyc1",
    #     "mith1": "/Users/matsuokoujirou/Documents/Data/Screening/Abnormal_data/mith1",
    #     "rbmp1": "/Users/matsuokoujirou/Documents/Data/Screening/Abnormal_data/rbmp1",
    #     "rbmp2": "/Users/matsuokoujirou/Documents/Data/Screening/Abnormal_data/rbmp2",
    #     "saga1": "/Users/matsuokoujirou/Documents/Data/Screening/Abnormal_data/saga1",
    #     "saga2": "/Users/matsuokoujirou/Documents/Data/Screening/Abnormal_data/saga2",   
    # }
    
    # 方法1-3を使用する場合（ファイルベース）
    if not test_files:
        print("No test files found. Please check the folder path or add files manually.")
        return
    output_dir = f"/Users/matsuokoujirou/Documents/Data/Screening/Results/{datetime.now().strftime('%Y%m%d_%H%M')}_file_based_screening"
    results, detailed_results = screener.screen_mutant_samples(test_files, output_dir)
    
    # 方法4を使用する場合（フォルダベース）
    # if not test_folders:
    #     print("No test folders found. Please check the folder paths.")
    #     return
    # output_dir = f"/Users/matsuokoujirou/Documents/Data/Screening/Results/{datetime.now().strftime('%Y%m%d_%H%M')}_folder_based_screening"
    # results, detailed_results = screener.screen_mutant_samples_by_folder(test_folders, output_dir)
    
    print(f"\n=== SCREENING COMPLETED ===")
    print(f"Results saved to: {output_dir}")
    
    # 簡易サマリー表示
    print(f"\nQUICK SUMMARY:")
    print(f"{'File':<15} {'Cells':<8} {'Conservative':<12} {'Moderate':<12} {'Status':<10}")
    print("-" * 70)
    
    # WT閾値を取得（結果から計算）
    wt_conservative = None
    for sample_name, result in results.items():
        if result.get('is_wt', False):
            wt_conservative = result['conservative_anomaly_rate'] * 100
            break
    
    if wt_conservative is not None:
        threshold = wt_conservative + 4.2
        print(f"WT Baseline: {wt_conservative:.1f}% | Anomaly Threshold: {threshold:.1f}%")
        print("-" * 70)
        
        for sample_name, result in results.items():
            status = "ANOMALY" if result['conservative_anomaly_rate']*100 > threshold else "NORMAL"
            if result.get('is_wt', False):
                status = "WT"
            
            print(f"{sample_name:<15} {result['total_cells']:<8} "
                  f"{result['conservative_anomaly_rate']*100:>8.1f}% "
                  f"{result['moderate_anomaly_rate']*100:>10.1f}% "
                  f"{status:<10}")
    else:
        for sample_name, result in results.items():
            print(f"{sample_name:<15} {result['total_cells']:<8} "
                  f"{result['conservative_anomaly_rate']*100:>8.1f}% "
              f"{result['moderate_anomaly_rate']*100:>10.1f}%")


if __name__ == "__main__":
    main()