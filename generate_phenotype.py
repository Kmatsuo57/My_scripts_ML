import numpy as np
import tifffile as tiff
import os
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from stardist.models import StarDist2D
from csbdeep.utils import normalize
from skimage.measure import regionprops
from skimage.transform import resize
# exposure は削除（コントラスト強調をしないため）

class RawPhenotypeVisualizer:
    def __init__(self, csv_path, folders_dict):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        self.df = pd.read_csv(csv_path)
        self.folders = folders_dict
        
        print("Loading StarDist model...")
        try:
            self.stardist_model = StarDist2D.from_pretrained('2D_versatile_fluo')
        except:
            self.stardist_model = StarDist2D.from_pretrained('2D_versatile_fluo')

    def extract_raw_cell(self, file_path, target_bbox):
        """
        コントラスト強調を行わず、生の輝度値で画像を抽出する
        target_bbox: (minr, minc, maxr, maxc)
        """
        try:
            image = tiff.imread(file_path)
            if image.ndim == 3 and image.shape[-1] >= 3:
                green_channel = image[..., 1]
            else:
                green_channel = image
            
            minr, minc, maxr, maxc = target_bbox
            cell_image = green_channel[minr:maxr, minc:maxc]
            
            # リサイズのみ行う（輝度はいじらない）
            # 視認性のため、リサイズ時は範囲を保持するように注意
            cell_image_resized = resize(cell_image, (64, 64), anti_aliasing=True, preserve_range=True)
            
            return cell_image_resized
        except Exception as e:
            print(f"Error extracting raw cell: {e}")
            return None

    def get_cell_info_and_image(self, sample_name, cell_id):
        # 該当する行を探す
        sample_df = self.df[self.df['sample_name'] == sample_name]
        target_row = sample_df[sample_df['cell_id'] == cell_id]
        
        if target_row.empty:
            return None, None
            
        file_name = target_row.iloc[0]['file_name']
        folder_path = self.folders.get(sample_name)
        if not folder_path: return None, None
        
        file_path = os.path.join(folder_path, file_name)
        
        # StarDistで再度検出して、同じBBoxを見つける必要がある
        # （CSVにBBoxを保存していないため、再計算が必要）
        try:
            image = tiff.imread(file_path)
            if image.ndim == 3 and image.shape[-1] >= 3:
                seg_channel = image[..., 2]
            else:
                seg_channel = image
            
            normalized_seg = normalize(seg_channel)
            labels, _ = self.stardist_model.predict_instances(normalized_seg)
            props = regionprops(labels)
            
            # フィルタリングロジック（厳密に合わせる）
            height, width = labels.shape
            valid_props = []
            for prop in props:
                minr, minc, maxr, maxc = prop.bbox
                if (minr < 10 or minc < 10 or maxr > (height - 10) or maxc > (width - 10)): continue
                if prop.area < 200 or prop.area > 8000: continue
                if prop.eccentricity > 0.95: continue
                
                # 強度チェック用の画像取得
                if image.ndim == 3 and image.shape[-1] >= 3:
                    gc = image[..., 1]
                else:
                    gc = image
                cell_img = gc[minr:maxr, minc:maxc]
                if np.mean(cell_img) < 0.5 or np.std(cell_img) < 0.1: continue
                
                valid_props.append(prop)
            
            # 該当するcell_idのプロパティを取得
            # 注: フォルダ全体の通し番号ではなく、ファイル内のローカルな順序で特定する必要がある
            # ここでは簡易的に「ファイル名とcell_id」が一致する行をCSVから特定したが、
            # CSVのcell_idは「フォルダ内の通し番号」であるため、ここでの特定は少し複雑。
            # 簡易化のため、「そのファイルに含まれる細胞」を上から順に取得し、
            # CSVのスコアと照らし合わせる方式は時間がかかるため、
            # ここでは「指定されたファイルを解析し、validな細胞を全て返す」方式をとる。
            
            # 修正: 上記のロジックは複雑なため、
            # 「各ファイルから抽出された細胞リスト」を生成し、そこから特定するアプローチをとる。
            return None, None # このメソッドは今回使わない実装に変更
            
        except:
            return None, None

    def create_raw_mosaic(self, target_samples, wt_sample='WT', top_n=5, output_dir='.'):
        print("Generating RAW image mosaic (No contrast enhancement)...")
        
        # ターゲットのリスト
        all_samples = [wt_sample] + target_samples
        
        n_rows = len(all_samples)
        n_cols = top_n
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.8))
        if n_rows == 1: axes = np.array([axes])
        if n_cols == 1: axes = np.array([[ax] for ax in axes])
        elif axes.ndim == 1: axes = axes.reshape(n_rows, n_cols)

        for r, sample in enumerate(all_samples):
            # サンプルのデータを取得
            s_df = self.df[self.df['sample_name'] == sample]
            
            if sample == wt_sample:
                # WTはスコアが0に近いもの（正常）
                candidates = s_df.iloc[(s_df['conservative_score'] - 0).abs().argsort()].head(top_n)
                title_suffix = "(Typical)"
            else:
                # 変異株はスコアが高いもの（異常）
                candidates = s_df.sort_values('conservative_score', ascending=False).head(top_n)
                title_suffix = "(Top Anomaly)"
            
            # 画像を収集
            images = []
            scores = []
            
            # ファイルごとにグルーピングして効率化
            files_to_process = candidates['file_name'].unique()
            
            # 一時的に抽出した細胞を保持する辞書 {(filename, local_idx): image}
            extracted_cache = {}
            
            for fname in files_to_process:
                fpath = os.path.join(self.folders[sample], fname)
                try:
                    img = tiff.imread(fpath)
                    if img.ndim == 3 and img.shape[-1] >= 3:
                        g_chan = img[..., 1]
                        s_chan = img[..., 2]
                    else:
                        g_chan = img
                        s_chan = img
                        
                    norm_seg = normalize(s_chan)
                    labels, _ = self.stardist_model.predict_instances(norm_seg)
                    props = regionprops(labels)
                    h, w = labels.shape
                    
                    valid_idx = 0
                    for prop in props:
                        minr, minc, maxr, maxc = prop.bbox
                        if (minr < 10 or minc < 10 or maxr > (h - 10) or maxc > (w - 10)): continue
                        if prop.area < 200 or prop.area > 8000: continue
                        if prop.eccentricity > 0.95: continue
                        c_img = g_chan[minr:maxr, minc:maxc]
                        if np.mean(c_img) < 0.5 or np.std(c_img) < 0.1: continue
                        
                        # 生画像を保存
                        extracted_cache[(fname, valid_idx)] = c_img
                        valid_idx += 1
                except Exception as e:
                    print(f"Error reading {fname}: {e}")

            # 候補の細胞を表示用に並べる
            # 注: CSVのcell_idはフォルダ通し番号のため、ファイル名とファイル内インデックスへの変換が必要だが、
            # ここでは簡易的に「ファイル名」が一致するものをキャッシュから探す（厳密なIDマッチングは複雑なため省略）
            # 代わりに、候補行の「画像」を特定するために、スクリプト1のロジック（ファイル順スキャン）を模倣する必要がある。
            
            # 簡易策: 上記のキャッシュロジックだとIDずれが起きるため、
            # PhenotypeVisualizerの `get_specific_cells` ロジック（ファイル順スキャン）を再利用し、
            # ただし画像処理部分のみ「生画像」に変える。
            pass 

    # --- 簡易実装版: 既存クラスを継承してメソッドだけ上書き ---
    # これが一番確実で早いです。

def main():
    # パス設定
    result_csv_path = "/Users/matsuokoujirou/Documents/Data/Screening/Results/20251210_1310_folder_based_screening/detailed_cell_results.csv"
    test_folders = {
        "WT": "/Users/matsuokoujirou/Documents/Data/Screening/Pools/normal/RG2",
        "epyc1": "/Users/matsuokoujirou/Documents/Data/Screening/Abnormal_data/epyc1",
        "mith1": "/Users/matsuokoujirou/Documents/Data/Screening/Abnormal_data/mith1",
        "rbmp1": "/Users/matsuokoujirou/Documents/Data/Screening/Abnormal_data/rbmp1",
        "rbmp2": "/Users/matsuokoujirou/Documents/Data/Screening/Abnormal_data/rbmp2",
        "saga1": "/Users/matsuokoujirou/Documents/Data/Screening/Abnormal_data/saga1",
        "saga2": "/Users/matsuokoujirou/Documents/Data/Screening/Abnormal_data/saga2",   
    }

    if not os.path.exists(result_csv_path):
        print("CSV not found.")
        return

    # 前回のPhenotypeVisualizerクラスを読み込み（同じファイルに記述するか、importする）
    # ここでは、前回のスクリプトの `extract_cells_from_file` メソッドの中身だけを
    # 「コントラスト強調なし」に書き換えたクラスを定義して実行します。
    
    viz = RawPhenotypeVisualizer(result_csv_path, test_folders)
    viz.create_raw_mosaic(['rbmp1', 'saga2', 'epyc1'], 'WT', output_dir=os.path.dirname(result_csv_path))

# クラス定義（修正版）
class RawPhenotypeVisualizer:
    def __init__(self, csv_path, folders_dict):
        self.df = pd.read_csv(csv_path)
        self.folders = folders_dict
        print("Loading StarDist...")
        self.stardist_model = StarDist2D.from_pretrained('2D_versatile_fluo')

    def get_specific_cells_raw(self, sample_name, target_cell_ids):
        # フォルダ内のファイルを順にスキャンして、IDが一致する細胞の「生画像」を返す
        folder_path = self.folders[sample_name]
        tif_files = sorted(glob(os.path.join(folder_path, '*.tif')))
        found_images = {}
        current_global_idx = 0
        target_set = set(target_cell_ids)
        
        for file_path in tif_files:
            if not target_set: break
            
            # --- 細胞抽出処理 (RAW) ---
            try:
                image = tiff.imread(file_path)
                if image.ndim == 3 and image.shape[-1] >= 3:
                    seg_c = image[..., 2]
                    green_c = image[..., 1]
                else:
                    seg_c = image
                    green_c = image
                
                # StarDist
                norm_seg = normalize(seg_c)
                labels, _ = self.stardist_model.predict_instances(norm_seg)
                props = regionprops(labels)
                h, w = labels.shape
                
                cells_in_file = []
                for prop in props:
                    minr, minc, maxr, maxc = prop.bbox
                    if (minr < 10 or minc < 10 or maxr > (h - 10) or maxc > (w - 10)): continue
                    if prop.area < 200 or prop.area > 8000: continue
                    if prop.eccentricity > 0.95: continue
                    
                    cell_img = green_c[minr:maxr, minc:maxc]
                    if np.mean(cell_img) < 0.5 or np.std(cell_img) < 0.1: continue
                    
                    # ★ここが変更点：コントラスト強調(equalize_adapthist)をしない！
                    # 単に表示用にリサイズだけする
                    cell_resized = resize(cell_img, (64, 64), anti_aliasing=True, preserve_range=True)
                    
                    # 表示用に正規化だけしておく (0-1) だが、極端な強調はしない
                    # 元の画素値の最大値で割る
                    img_max = np.max(green_c)
                    if img_max > 0:
                        cell_resized = cell_resized / img_max
                    
                    cells_in_file.append(cell_resized)
                
                # IDマッチング
                file_end_idx = current_global_idx + len(cells_in_file)
                matched_ids = [tid for tid in target_set if current_global_idx <= tid < file_end_idx]
                
                for tid in matched_ids:
                    local_idx = tid - current_global_idx
                    found_images[tid] = cells_in_file[local_idx]
                    target_set.remove(tid)
                
                current_global_idx += len(cells_in_file)
                
            except Exception as e:
                print(f"Error: {e}")
                continue
                
        return found_images

    def create_raw_mosaic(self, target_samples, wt_sample, output_dir):
        print("Generating RAW mosaic...")
        all_samples = [wt_sample] + target_samples
        n_rows = len(all_samples)
        top_n = 5
        
        fig, axes = plt.subplots(n_rows, top_n, figsize=(12, n_rows*2.5))
        if n_rows == 1: axes = np.array([axes])
        
        for r, sample in enumerate(all_samples):
            s_df = self.df[self.df['sample_name'] == sample]
            if sample == wt_sample:
                # WT: スコア0に近い
                cand = s_df.iloc[(s_df['conservative_score'] - 0).abs().argsort()].head(top_n)
            else:
                # Mutant: スコアが高い
                cand = s_df.sort_values('conservative_score', ascending=False).head(top_n)
            
            target_ids = cand['cell_id'].tolist()
            images_map = self.get_specific_cells_raw(sample, target_ids)
            
            scores = cand['conservative_score'].tolist()
            
            # Plot
            # 左端に行タイトル
            axes[r, 0].text(-0.3, 0.5, sample, transform=axes[r, 0].transAxes, va='center', ha='right', fontsize=12, fontweight='bold')
            
            img_list = [images_map.get(tid) for tid in target_ids]
            
            for c in range(top_n):
                ax = axes[r, c]
                if c < len(img_list) and img_list[c] is not None:
                    ax.imshow(img_list[c], cmap='gray', vmin=0, vmax=1) # 0-1で表示
                    ax.set_title(f"{scores[c]:.1f}", color='red' if scores[c]>13 else 'black')
                ax.axis('off')
        
        plt.suptitle("RAW Images (No Contrast Enhancement)", fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'phenotype_mosaic_raw.png'), dpi=300)
        plt.show()

if __name__ == "__main__":
    main()