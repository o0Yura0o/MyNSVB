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

def get_split_intervals(y, sr, min_dur=3.0, max_dur=8.2, top_db=30, max_sil_keep=1.0):
    """
    改良版切片演算法：
    1. 忽略過長的靜音間隔 (防止開頭噪音把靜音捲入)
    2. 捨棄過短且無法合併的片段
    """
    # 檢測非靜音區間
    intervals = librosa.effects.split(y, top_db=top_db, frame_length=2048, hop_length=512)
    
    final_intervals = []
    if len(intervals) == 0:
        return []

    # 初始化當前候選片段
    curr_start, curr_end = intervals[0]
    
    for next_start, next_end in intervals[1:]:
        curr_dur = (curr_end - curr_start) / sr
        gap_dur = (next_start - curr_end) / sr
        next_dur = (next_end - next_start) / sr
        
        # 判斷是否合併
        # 條件：總長度不超過 max_dur 且 中間靜音不超過 max_sil_keep
        if (curr_dur + gap_dur + next_dur <= max_dur) and (gap_dur <= max_sil_keep):
            # 合併：延伸結束點
            curr_end = next_end
        else:
            # 不合併：結算當前片段
            if (curr_end - curr_start) / sr >= min_dur:
                final_intervals.append((curr_start, curr_end))
            
            # 開始新的片段
            curr_start = next_start
            curr_end = next_end
            
    # 檢查最後一段
    if (curr_end - curr_start) / sr >= min_dur:
        final_intervals.append((curr_start, curr_end))
        
    return final_intervals

def save_slice(y, start, end, sr, path, pad_sec=0.1):
    pad = int(pad_sec * sr)
    s = max(0, start - pad)
    e = min(len(y), end + pad)
    chunk = y[s:e]
    sf.write(path, chunk, sr)

def process_pair(src_path, tgt_path, output_dir, top_db, target_sr=22050):
    filename = os.path.basename(src_path).split('.')[0]
    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing pair: {src_path} & {tgt_path}")

    # 1. 載入與格式統一
    try:
        y_src, _ = librosa.load(src_path, sr=target_sr, mono=True)
        y_tgt, _ = librosa.load(tgt_path, sr=target_sr, mono=True)
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # 2. 高通濾波
    y_src = highpass_filter(y_src, 50, target_sr)
    y_tgt = highpass_filter(y_tgt, 50, target_sr)

    # 3. 降噪
    if HAS_NOISEREDUCE:
        print("  -> Denoising...")
        y_src = apply_denoise(y_src, target_sr)
        y_tgt = apply_denoise(y_tgt, target_sr)

    # 4. 響度標準化
    print("  -> Normalizing Loudness...")
    y_src = normalize_loudness(y_src, target_sr)
    y_tgt = normalize_loudness(y_tgt, target_sr)

    # 5. 獨立切片
    print(f"  -> Slicing (top_db={top_db})...")
    # max_sil_keep=1.0: 如果靜音超過1秒，強制斷開，避免將前面的噪音與後面的歌聲合併
    intervals_src = get_split_intervals(y_src, target_sr, top_db=top_db, max_sil_keep=1.0)
    intervals_tgt = get_split_intervals(y_tgt, target_sr, top_db=top_db, max_sil_keep=1.0)
    
    # 6. 儲存
    max_idx = max(len(intervals_src), len(intervals_tgt))
    
    for i in range(max_idx):
        # 處理 Source
        if i < len(intervals_src):
            s, e = intervals_src[i]
            out_name = os.path.join(output_dir, f"{filename}_{i:03d}_A.wav")
            save_slice(y_src, s, e, target_sr, out_name)
        
        # 處理 Target
        if i < len(intervals_tgt):
            s, e = intervals_tgt[i]
            out_name = os.path.join(output_dir, f"{filename}_{i:03d}_P.wav")
            save_slice(y_tgt, s, e, target_sr, out_name)

    print(f"  -> Done! Source slices: {len(intervals_src)}, Target slices: {len(intervals_tgt)}")
    if len(intervals_src) != len(intervals_tgt):
        print(f"  [Info] Slice counts differ. Please check pairs manually.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="NeuralSVB Data Preprocessor")
    parser.add_argument('--src', type=str, required=True, help='Amateur audio path')
    parser.add_argument('--tgt', type=str, required=True, help='Professional audio path')
    parser.add_argument('--out_dir', type=str, default='processed_data', help='Output directory')
    parser.add_argument('--top_db', type=int, default=30, help='Silence threshold (higher = more sensitive to silence)')
    args = parser.parse_args()

    process_pair(args.src, args.tgt, args.out_dir, args.top_db)