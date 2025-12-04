import argparse
import os
import numpy as np
import librosa
import soundfile as sf
import pyloudnorm as pyln
from scipy.signal import butter, sosfilt
import warnings

warnings.filterwarnings("ignore")

# 嘗試導入 noisereduce
try:
    import noisereduce as nr
    HAS_NOISEREDUCE = True
except ImportError:
    HAS_NOISEREDUCE = False
    print("Warning: 'noisereduce' not found. Denoising will be skipped.")

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    sos = butter(order, normal_cutoff, btype='high', analog=False, output='sos')
    return sos

def highpass_filter(data, cutoff, fs, order=5):
    sos = butter_highpass(cutoff, fs, order=order)
    return sosfilt(sos, data)

def apply_denoise(y, sr):
    if HAS_NOISEREDUCE:
        try:
            return nr.reduce_noise(y=y, sr=sr, stationary=True, prop_decrease=0.95)
        except:
            return y
    return y

def normalize_loudness(y, sr, target_db=-22.0):
    meter = pyln.Meter(sr)
    try:
        loudness = meter.integrated_loudness(y)
        if np.isinf(loudness) or loudness < -70:
            return y
        y_norm = pyln.normalize.loudness(y, loudness, target_db)
        if np.max(np.abs(y_norm)) > 0.95:
            y_norm = y_norm / np.max(np.abs(y_norm)) * 0.95
        return y_norm
    except:
        return y

def get_intervals_from_silence(y, sr, min_dur=3.0, max_dur=9, top_db=30, max_sil_keep=1.0):
    """
    僅對基準音檔 (Target) 進行靜音檢測
    """
    raw_intervals = librosa.effects.split(y, top_db=top_db, frame_length=2048, hop_length=512)
    
    if len(raw_intervals) == 0: return []

    # 簡單合併邏輯
    merged = []
    curr_start, curr_end = raw_intervals[0]

    for next_start, next_end in raw_intervals[1:]:
        gap = (next_start - curr_end) / sr
        curr_len = (curr_end - curr_start) / sr
        next_len = (next_end - next_start) / sr
        
        # 如果合併後不超過 max_dur，且中間間隔不大，則合併
        if (curr_len + gap + next_len <= max_dur) and (gap <= max_sil_keep):
            curr_end = next_end
        else:
            if (curr_end - curr_start) / sr >= min_dur:
                merged.append((curr_start, curr_end))
            curr_start = next_start
            curr_end = next_end
            
    if (curr_end - curr_start) / sr >= 1.0:
        merged.append((curr_start, curr_end))
        
    return merged

def map_intervals_via_dtw(y_src, y_tgt, sr, intervals_tgt):
    """
    使用 DTW 將 Target 的切片時間點映射到 Source
    """
    print("  -> Calculating DTW alignment (this may take a moment)...")
    
    # 1. 提取 Chroma 特徵 (對音色不敏感，對旋律敏感)
    # 使用 CENS 特徵以獲得更好的長度魯棒性
    chroma_src = librosa.feature.chroma_cens(y=y_src, sr=sr, hop_length=512)
    chroma_tgt = librosa.feature.chroma_cens(y=y_tgt, sr=sr, hop_length=512)
    
    # 2. 計算 DTW 路徑
    # D: 距離矩陣, wp: 路徑 [(src_idx, tgt_idx), ...]
    # subseq=True 允許局部對齊，但這裡我們假設是整首對整首，用全局對齊
    D, wp = librosa.sequence.dtw(X=chroma_src, Y=chroma_tgt, metric='cosine')
    
    # wp 是倒序的，轉正並轉為 numpy array
    wp = wp[::-1] 
    wp_src = wp[:, 0] # Source 幀索引
    wp_tgt = wp[:, 1] # Target 幀索引
    
    intervals_src = []
    
    # 3. 映射時間點
    for t_start_sample, t_end_sample in intervals_tgt:
        # 將 Sample 轉為 Frame 索引
        t_start_frame = librosa.samples_to_frames(t_start_sample, hop_length=512)
        t_end_frame = librosa.samples_to_frames(t_end_sample, hop_length=512)
        
        # 在路徑中找到最接近 Target Frame 的點
        # np.searchsorted 或是 argmin
        idx_start = np.argmin(np.abs(wp_tgt - t_start_frame))
        idx_end = np.argmin(np.abs(wp_tgt - t_end_frame))
        
        # 找到對應的 Source Frame
        s_start_frame = wp_src[idx_start]
        s_end_frame = wp_src[idx_end]
        
        # 轉回 Sample
        s_start_sample = librosa.frames_to_samples(s_start_frame, hop_length=512)
        s_end_sample = librosa.frames_to_samples(s_end_frame, hop_length=512)
        
        intervals_src.append((s_start_sample, s_end_sample))
        
    return intervals_src

def save_slice(y, start, end, sr, path, pad_sec=0.2):
    pad = int(pad_sec * sr)
    s = max(0, start - pad)
    e = min(len(y), end + pad)
    chunk = y[s:e]
    sf.write(path, chunk, sr)

def process_pair_aligned(src_path, tgt_path, output_dir, top_db, target_sr=22050):
    filename = os.path.basename(src_path).split('.')[0]
    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing pair (Aligned Slicing): {src_path} & {tgt_path}")

    # 1. 載入
    try:
        y_src, _ = librosa.load(src_path, sr=target_sr, mono=True)
        y_tgt, _ = librosa.load(tgt_path, sr=target_sr, mono=True)
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # 2. 預處理 (濾波、降噪、響度)
    y_src = highpass_filter(y_src, 50, target_sr)
    y_tgt = highpass_filter(y_tgt, 50, target_sr)

    if HAS_NOISEREDUCE:
        print("  -> Denoising...")
        y_src = apply_denoise(y_src, target_sr)
        y_tgt = apply_denoise(y_tgt, target_sr)

    print("  -> Normalizing Loudness...")
    y_src = normalize_loudness(y_src, target_sr)
    y_tgt = normalize_loudness(y_tgt, target_sr)

    # 3. 僅對 Target 進行切片分析
    print(f"  -> Analyzing Target structure (top_db={top_db})...")
    intervals_tgt = get_intervals_from_silence(y_tgt, target_sr, top_db=top_db)
    
    if not intervals_tgt:
        print("  [Error] No intervals found in target audio.")
        return

    # 4. 使用 DTW 映射到 Source
    print(f"  -> Mapping {len(intervals_tgt)} slices to Source via DTW...")
    intervals_src = map_intervals_via_dtw(y_src, y_tgt, target_sr, intervals_tgt)
    
    # 5. 儲存 (保證一一對應)
    for i in range(len(intervals_tgt)):
        # Source
        s_src, e_src = intervals_src[i]
        out_name_src = os.path.join(output_dir, f"{filename}_{i:03d}_A.wav")
        save_slice(y_src, s_src, e_src, target_sr, out_name_src)
        
        # Target
        s_tgt, e_tgt = intervals_tgt[i]
        out_name_tgt = os.path.join(output_dir, f"{filename}_{i:03d}_P.wav")
        save_slice(y_tgt, s_tgt, e_tgt, target_sr, out_name_tgt)

    print(f"  -> Done! Generated {len(intervals_tgt)} aligned pairs in '{output_dir}'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="NeuralSVB Data Preprocessor (Aligned Slicing)")
    parser.add_argument('--src', type=str, required=True, help='Amateur audio path')
    parser.add_argument('--tgt', type=str, required=True, help='Professional audio path')
    parser.add_argument('--out_dir', type=str, default='processed_data', help='Output directory')
    parser.add_argument('--top_db', type=int, default=30, help='Silence threshold for Target audio')
    args = parser.parse_args()

    process_pair_aligned(args.src, args.tgt, args.out_dir, args.top_db)