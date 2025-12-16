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

# 【重要】キャリブレーション用に必要なscikit-learnライブラリ
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

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
        
        # AutoencoderとEncoder（これらは不変の特徴抽出器として使用）
        self.autoencoder = load_model(os.path.join(self.model_dir, 'best_autoencoder.keras'))
        self.encoder = load_model(os.path.join(self.model_dir, 'encoder.keras'))
        
        # デフォルトの前処理器と検知器（キャリブレーション失敗時やWTなしの場合のバックアップ）
        with open(os.path.join(self.model_dir, 'scaler.pkl'), 'rb') as f:
            self.scaler = pickle.load(f)
        with open(os.path.join(self.model_dir, 'pca.pkl'), 'rb') as f:
            self.pca = pickle.load(f)
        
        with open(os.path.join(self.model_dir, 'detector_conservative.pkl'), 'rb') as f:
            self.detector_conservative = pickle.load(f)
        with open(os.path.join(self.model_dir, 'detector_moderate.pkl'), 'rb') as f:
            self.detector_moderate = pickle.load(f)
        
        # StarDist
        self.stardist_model = StarDist2D.from_pretrained('2D_versatile_fluo')
        
        print("All models loaded successfully!")
    
    def extract_quality_cells(self, image_path):
        """品質管理付き細胞抽出"""
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
            
            # 品質フィルタリング
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
                
                # 前処理（ヒストグラム均等化＋リサイズ）
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

    def calibrate_on_current_wt(self, wt_file_path):
        """
        【重要】当日のWTファイルを使って異常検知器（Scaler, PCA, SVM）を再学習する。
        これにより、日差（バッチ効果）による誤検知を防ぐ。
        """
        print(f"\n=== Calibrating Detectors on Current WT: {os.path.basename(wt_file_path)} ===")
        
        # 1. WT細胞の抽出
        cells, _ = self.extract_quality_cells(wt_file_path)
        
        # 細胞数が少なすぎる場合は警告して中断
        if len(cells) < 30:
            print(f"[WARNING] Only {len(cells)} cells found in WT. Need at least 30 for reliable calibration.")
            print("Using pre-trained detectors instead (Warning: Batch effects may occur).")
            return False
            
        print(f"  Training local detector with {len(cells)} WT cells...")
        
        # 2. 特徴抽出（学習済みのEncoderを使用）
        # Encoderは「形を見る目」なので、再学習せずそのまま使う
        X = np.expand_dims(np.array(cells), axis=-1).astype('float32')
        encoded_features = self.encoder.predict(X, verbose=0)
        encoded_flat = encoded_features.reshape(len(encoded_features), -1)
        
        # 3. 前処理とPCAの再適合
        # ここで「今日のデータの分布」に合わせてスケーリング基準を作り直す
        self.scaler = RobustScaler()
        features_scaled = self.scaler.fit_transform(encoded_flat)
        
        # PCAも今日のデータに合わせて主成分軸を引き直す
        n_components = min(100, features_scaled.shape[1], features_scaled.shape[0] - 1)
        self.pca = PCA(n_components=n_components)
        features_reduced = self.pca.fit_transform(features_scaled)
        
        # 4. SVMの再学習 (OneClassSVM)
        # Conservative (nu=0.05: 下位5%を異常とみなす設定)
        self.detector_conservative = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
        self.detector_conservative.fit(features_reduced)
        
        # Moderate (nu=0.10: 下位10%を異常とみなす設定)
        self.detector_moderate = OneClassSVM(kernel='rbf', gamma='scale', nu=0.10)
        self.detector_moderate.fit(features_reduced)
        
        print("  Calibration complete. Detectors optimized for today's condition.")
        return True
    
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
        
        # 前処理（キャリブレーション済みのScaler/PCAを使用）
        encoded_scaled = self.scaler.transform(encoded_flat)
        encoded_pca = self.pca.transform(encoded_scaled)
        
        # 異常検知 (キャリブレーション済みのSVMを使用)
        conservative_predictions = self.detector_conservative.predict(encoded_pca)
        moderate_predictions = self.detector_moderate.predict(encoded_pca)
        
        conservative_scores = self.detector_conservative.decision_function(encoded_pca)
        moderate_scores = self.detector_moderate.decision_function(encoded_pca)
        
        return {
            'reconstruction_mse': mse_errors,
            'reconstruction_mae': mae_errors,
            'conservative_predictions': conservative_predictions,
            'moderate_predictions': moderate_predictions,
            'conservative_scores': -conservative_scores,
            'moderate_scores': -moderate_scores,
            'conservative_anomaly_rate': np.sum(conservative_predictions == -1) / len(conservative_predictions),
            'moderate_anomaly_rate': np.sum(moderate_predictions == -1) / len(moderate_predictions)
        }
    
    def screen_mutant_samples(self, test_files_dict, output_dir):
        """変異株スクリーニング実行（WT自動検出＆キャリブレーション機能付き）"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("=== Starting Mutant Screening with Calibration ===")
        
        # 1. WTファイルを探してキャリブレーションを実行
        print("\n--- Checking for WT file for calibration ---")
        wt_path = None
        # ファイル名に 'WT' が含まれるかチェック（大文字小文字区別なし）
        for name, path in test_files_dict.items():
            if 'WT' in name.upper().split('_') or name.upper() == 'WT':
                wt_path = path
                print(f"  Found WT file: {name}")
                break
        
        is_calibrated = False
        if wt_path:
            # ここでキャリブレーションを実行
            is_calibrated = self.calibrate_on_current_wt(wt_path)
        else:
            print("  [WARNING] WT file not found in the list. Using pre-trained models (Risk of Batch Effect).")
        
        results = {}
        detailed_results = []
        
        # WT閾値の設定（レポート用）
        # キャリブレーション後は、WTの異常率は理論上 5% と 10% に近くなる
        wt_thresholds = {'wt_conservative_rate': 5.0, 'wt_moderate_rate': 10.0, 
                         'threshold_conservative': 9.2, 'threshold_moderate': 14.2}
        
        if is_calibrated and wt_path:
             # 検証のため、キャリブレーション後のWT異常率を再計算
             cells, _ = self.extract_quality_cells(wt_path)
             scores = self.compute_anomaly_scores(cells)
             wt_rate_cons = scores['conservative_anomaly_rate'] * 100
             wt_rate_mod = scores['moderate_anomaly_rate'] * 100
             
             wt_thresholds = {
                 'wt_conservative_rate': wt_rate_cons,
                 'wt_moderate_rate': wt_rate_mod,
                 'threshold_conservative': wt_rate_cons + 4.2,  # WT + 4.2% を異常閾値とする
                 'threshold_moderate': wt_rate_mod + 4.2
             }
             print(f"  Post-calibration WT anomaly rate (Conservative): {wt_rate_cons:.2f}%")
        
        # 2. 全サンプルの解析ループ
        for sample_name, file_path in test_files_dict.items():
            print(f"\nProcessing {sample_name} ({os.path.basename(file_path)})...")
            
            if not os.path.exists(file_path):
                print(f"  File not found: {file_path}")
                continue
            
            # 細胞抽出
            cells, stats = self.extract_quality_cells(file_path)
            
            if len(cells) == 0:
                print(f"  No quality cells extracted from {sample_name}")
                continue
            
            # 異常スコア計算（キャリブレーション済みのモデルで判定）
            anomaly_scores = self.compute_anomaly_scores(cells)
            
            # 結果サマリー
            sample_result = {
                'sample_name': sample_name,
                'file_path': file_path,
                'total_cells': len(cells),
                'conservative_anomaly_rate': anomaly_scores['conservative_anomaly_rate'],
                'moderate_anomaly_rate': anomaly_scores['moderate_anomaly_rate'],
                'mean_mse': np.mean(anomaly_scores['reconstruction_mse']),
                'mean_mae': np.mean(anomaly_scores['reconstruction_mae']),
                'is_wt': sample_name.upper() == 'WT' or 'WT' in sample_name.upper()
            }
            
            results[sample_name] = sample_result
            
            # 詳細結果（細胞レベル）
            for i, (mse, mae, cons_pred, mod_pred) in enumerate(zip(
                anomaly_scores['reconstruction_mse'],
                anomaly_scores['reconstruction_mae'],
                anomaly_scores['conservative_predictions'],
                anomaly_scores['moderate_predictions']
            )):
                detailed_results.append({
                    'sample_name': sample_name,
                    'file_name': os.path.basename(file_path),
                    'cell_id': i,
                    'mse': mse,
                    'mae': mae,
                    'conservative_anomaly': cons_pred == -1,
                    'moderate_anomaly': mod_pred == -1
                })
            
            # 進捗表示
            print(f"    Cells: {len(cells)}")
            print(f"    Conservative anomaly rate: {sample_result['conservative_anomaly_rate']*100:.2f}%")
        
        # 3. 結果の保存と可視化
        if results:
            self.save_and_visualize_results(results, detailed_results, output_dir, wt_thresholds)
        
        return results, detailed_results
    
    def save_and_visualize_results(self, results, detailed_results, output_dir, wt_thresholds):
        """結果の保存と可視化（簡易版）"""
        # DataFrame化
        results_df = pd.DataFrame.from_dict(results, orient='index')
        results_df.to_csv(os.path.join(output_dir, 'screening_summary.csv'))
        
        detailed_df = pd.DataFrame(detailed_results)
        detailed_df.to_csv(os.path.join(output_dir, 'detailed_cell_results.csv'), index=False)
        
        # 1. 異常率の棒グラフ
        plt.figure(figsize=(12, 6))
        
        names = [name[:15] for name in results_df['sample_name']]
        rates = results_df['conservative_anomaly_rate'] * 100
        colors = ['blue' if row['is_wt'] else 'lightcoral' for _, row in results_df.iterrows()]
        
        plt.bar(names, rates, color=colors, alpha=0.7)
        
        # 閾値ライン
        plt.axhline(y=wt_thresholds['threshold_conservative'], color='red', linestyle='--', 
                   label=f"Threshold ({wt_thresholds['threshold_conservative']:.1f}%)")
        plt.axhline(y=wt_thresholds['wt_conservative_rate'], color='blue', linestyle=':', 
                   label=f"WT Baseline ({wt_thresholds['wt_conservative_rate']:.1f}%)")
        
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Anomaly Rate (%)')
        plt.title('Anomaly Rates (Calibrated on Today\'s WT)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'anomaly_rates_calibrated.png'), dpi=300)
        plt.show()
        
        # レポート生成
        self.generate_screening_report(results_df, output_dir, wt_thresholds)
        
    def generate_screening_report(self, results_df, output_dir, wt_thresholds):
        """レポート生成"""
        with open(os.path.join(output_dir, 'screening_report.txt'), 'w') as f:
            f.write("=== MUTANT SCREENING REPORT (CALIBRATED) ===\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("CALIBRATION INFO:\n")
            f.write(f"- WT Baseline set to: {wt_thresholds['wt_conservative_rate']:.2f}%\n")
            f.write(f"- Anomaly Threshold: {wt_thresholds['threshold_conservative']:.2f}%\n\n")
            
            f.write("RESULTS:\n")
            f.write(f"{'Sample':<20} {'Cells':<8} {'Anomaly Rate':<15} {'Status':<10}\n")
            f.write("-" * 60 + "\n")
            
            thresh = wt_thresholds['threshold_conservative']
            
            for _, row in results_df.iterrows():
                rate = row['conservative_anomaly_rate'] * 100
                status = "HIT" if rate > thresh else "NORMAL"
                if row['is_wt']: status = "WT"
                
                f.write(f"{row['sample_name']:<20} {row['total_cells']:<8} {rate:>10.1f}%    {status:<10}\n")


def main():
    # 設定
    model_dir = "/Users/matsuokoujirou/Documents/Data/imaging_data/Luca/results/20251216_1617" # モデルのパス
    
    # 解析したいフォルダのパス（ここに必ずWTの画像も含めてください！）
    folder_path = "/Users/matsuokoujirou/Documents/Data/imaging_data/Luca/Cells/All" 
    
    # スクリーナー初期化
    screener = ProductionMutantScreening(model_dir)
    
    # ファイル取得
    test_files = screener.get_files_from_folder(folder_path)
    
    if not test_files:
        print("No files found to process.")
        return

    # 出力先
    output_dir = f"/Users/matsuokoujirou/Documents/Data/imaging_data/Luca/{datetime.now().strftime('%Y%m%d_%H%M')}_calibrated"
    
    # 実行
    results, _ = screener.screen_mutant_samples(test_files, output_dir)
    
    print(f"\n=== Completed. Check results in {output_dir} ===")

if __name__ == "__main__":
    main()