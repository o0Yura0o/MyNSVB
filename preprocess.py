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
    """濾除低頻噪音"""
    sos = butter_highpass(cutoff, fs, order=order)
    y = sosfilt(sos, data)
    return y

def apply_denoise(y, sr):
    """去除平穩底噪"""
    if HAS_NOISEREDUCE:
        try:
            return nr.reduce_noise(y=y, sr=sr, stationary=True, prop_decrease=0.95)
        except Exception as e:
            print(f"Denoise failed: {e}")
            return y
    return y

def normalize_loudness(y, sr, target_db=-22.0):
    """統一響度至 target_db LUFS"""
    meter = pyln.Meter(sr)
    try:
        loudness = meter.integrated_loudness(y)
        if np.isinf(loudness) or loudness < -70:
            return y
        y_norm = pyln.normalize.loudness(y, loudness, target_db)
        if np.max(np.abs(y_norm)) > 0.95:
            y_norm = y_norm / np.max(np.abs(y_norm)) * 0.95
        return y_norm
    except Exception as e:
        return y

def merge_short_gaps(intervals, sr, min_gap_dur=0.5):
    """
    第一階段合併：
    強制合併間隔小於 min_gap_dur 的片段，防止句子中間因為換氣被切斷。
    """
    if len(intervals) == 0:
        return []

    merged = []
    curr_start, curr_end = intervals[0]

    for next_start, next_end in intervals[1:]:
        gap = (next_start - curr_end) / sr
        
        if gap < min_gap_dur:
            # 間隙太短，視為同一句，合併
            curr_end = next_end
        else:
            # 間隙夠長，結算上一句
            merged.append((curr_start, curr_end))
            curr_start = next_start
            curr_end = next_end
    
    merged.append((curr_start, curr_end))
    return merged

def force_split(start, end, sr, max_dur):
    """
    強制切分過長的片段
    如果一段聲音原始長度就超過 max_dur，將其切成多段
    """
    dur = (end - start) / sr
    if dur <= max_dur:
        return [(start, end)]
    
    splits = []
    curr = start
    while curr < end:
        # 嘗試切在 max_dur 處
        next_split = curr + int(max_dur * sr)
        
        # 如果剩下的一小段太短 (< 1.0s) 且不是最後一點點，就乾脆這一次切長一點包進來
        # 但如果太長還是得切
        if next_split >= end:
            splits.append((curr, end))
            break
        
        splits.append((curr, next_split))
        curr = next_split
        
    return splits

def get_split_intervals(y, sr, min_dur=3.0, max_dur=9.0, top_db=30, max_sil_keep=1.0, min_gap=0.5):
    """
    改良版切片演算法 (三階段：微觀合併 -> 宏觀合併 -> 強制切分)
    """
    # 1. 檢測非靜音區間
    raw_intervals = librosa.effects.split(y, top_db=top_db, frame_length=2048, hop_length=512)
    
    # 2. 第一階段：微觀合併 (防止斷句)
    intervals = merge_short_gaps(raw_intervals, sr, min_gap_dur=min_gap)
    
    if len(intervals) == 0:
        return []

    # 3. 第二階段：宏觀合併 (長度控制)
    merged_intervals = []
    curr_start, curr_end = intervals[0]
    
    for next_start, next_end in intervals[1:]:
        curr_dur = (curr_end - curr_start) / sr
        gap_dur = (next_start - curr_end) / sr
        next_dur = (next_end - next_start) / sr
        
        total_dur_if_merge = curr_dur + gap_dur + next_dur

        # 修正後的合併邏輯：
        # 只有在「合併後總長度小於 max_dur」的前提下，才考慮其他條件
        # 條件 A: 總長度OK 且 (當前太短 或 中間靜音很短)
        can_merge_len = total_dur_if_merge <= max_dur
        should_merge_logic = (curr_dur < min_dur) or (gap_dur <= max_sil_keep)
        
        if can_merge_len and should_merge_logic:
            # 合併
            curr_end = next_end
        else:
            # 結算並檢查是否需要強制切分
            sub_intervals = force_split(curr_start, curr_end, sr, max_dur)
            # 過濾掉切分後太短的碎片 (例如 < 0.5s)，除非它是唯一的
            for s, e in sub_intervals:
                if (e - s)/sr >= 1.0:
                    merged_intervals.append((s, e))
            
            curr_start = next_start
            curr_end = next_end
            
    # 最後一段
    sub_intervals = force_split(curr_start, curr_end, sr, max_dur)
    for s, e in sub_intervals:
        if (e - s)/sr >= 1.0:
            merged_intervals.append((s, e))
        
    return merged_intervals

def save_slice(y, start, end, sr, path, pad_sec=0.3):
    pad = int(pad_sec * sr)
    s = max(0, start - pad)
    e = min(len(y), end + pad)
    chunk = y[s:e]
    sf.write(path, chunk, sr)

def process_pair(src_path, tgt_path, output_dir, top_db, target_sr=22050):
    filename = os.path.basename(src_path).split('.')[0]
    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing pair: {src_path} & {tgt_path}")

    try:
        y_src, _ = librosa.load(src_path, sr=target_sr, mono=True)
        y_tgt, _ = librosa.load(tgt_path, sr=target_sr, mono=True)
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # 預處理
    y_src = highpass_filter(y_src, 50, target_sr)
    y_tgt = highpass_filter(y_tgt, 50, target_sr)

    if HAS_NOISEREDUCE:
        print("  -> Denoising...")
        y_src = apply_denoise(y_src, target_sr)
        y_tgt = apply_denoise(y_tgt, target_sr)

    print("  -> Normalizing Loudness...")
    y_src = normalize_loudness(y_src, target_sr)
    y_tgt = normalize_loudness(y_tgt, target_sr)

    # 切片
    print(f"  -> Slicing (top_db={top_db})...")
    intervals_src = get_split_intervals(y_src, target_sr, top_db=top_db, max_sil_keep=1.0, min_gap=0.5)
    intervals_tgt = get_split_intervals(y_tgt, target_sr, top_db=top_db, max_sil_keep=1.0, min_gap=0.5)
    
    # 儲存
    max_idx = max(len(intervals_src), len(intervals_tgt))
    
    for i in range(max_idx):
        if i < len(intervals_src):
            s, e = intervals_src[i]
            out_name = os.path.join(output_dir, f"{filename}_{i:03d}_A.wav")
            save_slice(y_src, s, e, target_sr, out_name)
        
        if i < len(intervals_tgt):
            s, e = intervals_tgt[i]
            out_name = os.path.join(output_dir, f"{filename}_{i:03d}_P.wav")
            save_slice(y_tgt, s, e, target_sr, out_name)

    print(f"  -> Done! Source slices: {len(intervals_src)}, Target slices: {len(intervals_tgt)}")
    if len(intervals_src) != len(intervals_tgt):
        print(f"  [Warning] Slice counts differ. Source: {len(intervals_src)} vs Target: {len(intervals_tgt)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="NeuralSVB Data Preprocessor (Robust Slicing)")
    parser.add_argument('--src', type=str, required=True, help='Amateur audio path')
    parser.add_argument('--tgt', type=str, required=True, help='Professional audio path')
    parser.add_argument('--out_dir', type=str, default='processed_data', help='Output directory')
    parser.add_argument('--top_db', type=int, default=30, help='Silence threshold (higher = more sensitive)')
    args = parser.parse_args()

    process_pair(args.src, args.tgt, args.out_dir, args.top_db)