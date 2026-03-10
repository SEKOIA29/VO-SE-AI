"""
VO-SE Pro - AuralAI Model Trainer
oto.ini付きUTAU音源から学習データを生成し、
aural_dynamics.onnx を出力するスクリプト。

使い方:
  1. assets/training_voices/ 以下にUTAU音源フォルダを置く
     (oto.ini と WAVファイルが入っていること)
  2. pip install numpy scipy scikit-learn onnx skl2onnx tqdm librosa
  3. python train_aural_model.py
  4. models/aural_dynamics.onnx が生成される
"""

import os
import glob
import wave
import struct
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# ─────────────────────────────────────────────
# 0. 設定
# ─────────────────────────────────────────────
VOICE_DIR   = "assets/training_voices"   # UTAU音源のルートフォルダ
OUTPUT_DIR  = "models"                   # ONNXの出力先
OUTPUT_NAME = "aural_dynamics.onnx"
TARGET_FS   = 44100
N_MFCC      = 20     # MFCC次元数
N_FRAMES    = 64     # 1サンプルあたりのフレーム数（固定長パディング）
MIN_SAMPLES = 30     # 学習に必要な最低サンプル数

# ─────────────────────────────────────────────
# 1. oto.ini パーサー
# ─────────────────────────────────────────────
@dataclass
class OtoEntry:
    wav_path:       str
    alias:          str
    offset:         float   # ms
    consonant:      float   # ms
    cutoff:         float   # ms (負なら末尾からの距離)
    pre_utterance:  float   # ms
    overlap:        float   # ms

def parse_oto_ini(oto_path: str) -> List[OtoEntry]:
    """oto.ini を読み込んでエントリリストを返す。cp932/utf-8 両対応。"""
    entries = []
    voice_dir = os.path.dirname(oto_path)

    for enc in ("cp932", "utf-8", "utf-8-sig"):
        try:
            with open(oto_path, encoding=enc) as f:
                lines = f.readlines()
            break
        except (UnicodeDecodeError, FileNotFoundError):
            continue
    else:
        print(f"  [WARN] Cannot read: {oto_path}")
        return entries

    for line in lines:
        line = line.strip()
        if not line or line.startswith(";") or "=" not in line:
            continue
        try:
            wav_name, params_str = line.split("=", 1)
            parts = params_str.split(",")
            if len(parts) < 5:
                continue
            alias       = parts[0].strip()
            offset      = float(parts[1]) if parts[1].strip() else 0.0
            consonant   = float(parts[2]) if parts[2].strip() else 0.0
            cutoff      = float(parts[3]) if parts[3].strip() else 0.0
            pre_utt     = float(parts[4]) if parts[4].strip() else 0.0
            overlap     = float(parts[5]) if len(parts) > 5 and parts[5].strip() else 0.0

            wav_path = os.path.join(voice_dir, wav_name.strip())
            if not wav_path.endswith(".wav"):
                wav_path += ".wav"

            entries.append(OtoEntry(
                wav_path=wav_path,
                alias=alias,
                offset=offset,
                consonant=consonant,
                cutoff=cutoff,
                pre_utterance=pre_utt,
                overlap=overlap,
            ))
        except (ValueError, IndexError):
            continue

    return entries

# ─────────────────────────────────────────────
# 2. WAV読み込み & リサンプリング
# ─────────────────────────────────────────────
def load_wav(path: str) -> Optional[np.ndarray]:
    """WAVをfloat32モノラルで返す。失敗時はNone。"""
    try:
        with wave.open(path, "rb") as wf:
            fs      = wf.getframerate()
            nch     = wf.getnchannels()
            nframes = wf.getnframes()
            raw     = wf.readframes(nframes)

        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

        # ステレオ→モノラル
        if nch == 2:
            data = data[::2]

        # リサンプリング（簡易線形補間）
        if fs != TARGET_FS:
            from math import gcd
            g = gcd(fs, TARGET_FS)
            try:
                from scipy.signal import resample_poly
                data = resample_poly(data, TARGET_FS // g, fs // g).astype(np.float32)
            except ImportError:
                # scipy がなければ線形補間で代替
                n_new = int(len(data) * TARGET_FS / fs)
                data = np.interp(
                    np.linspace(0, len(data) - 1, n_new),
                    np.arange(len(data)),
                    data
                ).astype(np.float32)

        return data
    except Exception as e:
        return None

# ─────────────────────────────────────────────
# 3. MFCC抽出（librosaなし純NumPy版 & librosa版）
# ─────────────────────────────────────────────
def _stft_numpy(signal: np.ndarray, n_fft=512, hop=256) -> np.ndarray:
    """純NumPyによる簡易STFT（絶対値スペクトル）。"""
    window = np.hanning(n_fft)
    frames = []
    for i in range(0, len(signal) - n_fft, hop):
        frame = signal[i:i + n_fft] * window
        frames.append(np.abs(np.fft.rfft(frame)))
    return np.array(frames).T if frames else np.zeros((n_fft // 2 + 1, 1))

def extract_mfcc_numpy(signal: np.ndarray, n_mfcc=N_MFCC, sr=TARGET_FS) -> np.ndarray:
    """純NumPyでMFCCを計算する（librosa非依存）。"""
    n_fft = 512
    hop   = 256
    n_mels = 40

    spec = _stft_numpy(signal, n_fft, hop)  # (freq_bins, frames)

    # メルフィルタバンク
    f_min, f_max = 0.0, sr / 2.0
    mel_min = 2595 * np.log10(1 + f_min / 700)
    mel_max = 2595 * np.log10(1 + f_max / 700)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    freq_points = 700 * (10 ** (mel_points / 2595) - 1)
    bin_points = np.floor((n_fft + 1) * freq_points / sr).astype(int)

    fbank = np.zeros((n_mels, spec.shape[0]))
    for m in range(1, n_mels + 1):
        f_m_minus = bin_points[m - 1]
        f_m       = bin_points[m]
        f_m_plus  = bin_points[m + 1]
        for k in range(f_m_minus, f_m):
            if f_m != f_m_minus:
                fbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            if f_m_plus != f_m:
                fbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

    mel_spec = np.dot(fbank, spec)
    log_mel  = np.log(mel_spec + 1e-8)

    # DCT
    dct_mat = np.cos(np.pi * np.outer(np.arange(n_mfcc), np.arange(1, n_mels + 1) - 0.5) / n_mels)
    mfcc = np.dot(dct_mat, log_mel)  # (n_mfcc, frames)
    return mfcc

def extract_mfcc(signal: np.ndarray) -> np.ndarray:
    """MFCCを抽出。librosaがあれば使い、なければNumPy版にフォールバック。"""
    try:
        import librosa
        mfcc = librosa.feature.mfcc(y=signal, sr=TARGET_FS, n_mfcc=N_MFCC)
        return mfcc
    except ImportError:
        return extract_mfcc_numpy(signal)

def mfcc_to_fixed_length(mfcc: np.ndarray, n_frames=N_FRAMES) -> np.ndarray:
    """MFCCをN_FRAMES固定長にパディング/トリミングして平坦化。"""
    if mfcc.shape[1] >= n_frames:
        mfcc = mfcc[:, :n_frames]
    else:
        pad = np.zeros((mfcc.shape[0], n_frames - mfcc.shape[1]))
        mfcc = np.hstack([mfcc, pad])
    return mfcc.flatten().astype(np.float32)  # (N_MFCC * N_FRAMES,)

# ─────────────────────────────────────────────
# 4. 教師データ生成
# ─────────────────────────────────────────────
def build_dataset(voice_root: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    voice_root以下の全oto.iniを読んでデータセットを構築。
    Returns:
        X: (N, N_MFCC * N_FRAMES) float32
        y: (N, 3) float32  [pre_utterance, overlap, consonant] (ms正規化済み)
    """
    oto_files = glob.glob(os.path.join(voice_root, "**", "oto.ini"), recursive=True)
    print(f"Found {len(oto_files)} oto.ini files")

    X_list, y_list = [], []
    skipped = 0

    for oto_path in oto_files:
        entries = parse_oto_ini(oto_path)
        print(f"  {oto_path}: {len(entries)} entries")

        for entry in entries:
            signal = load_wav(entry.wav_path)
            if signal is None or len(signal) < TARGET_FS * 0.05:
                skipped += 1
                continue

            # offset(ms)から切り出し開始位置を計算
            start = int(entry.offset * TARGET_FS / 1000)
            start = max(0, min(start, len(signal) - 1))
            segment = signal[start:]
            if len(segment) < 512:
                skipped += 1
                continue

            mfcc = extract_mfcc(segment)
            feat = mfcc_to_fixed_length(mfcc)

            # ラベル: ms値を0-1にソフト正規化（300ms上限）
            MAX_MS = 300.0
            label = np.array([
                np.clip(entry.pre_utterance / MAX_MS, 0.0, 1.0),
                np.clip(entry.overlap        / MAX_MS, 0.0, 1.0),
                np.clip(entry.consonant      / MAX_MS, 0.0, 1.0),
            ], dtype=np.float32)

            X_list.append(feat)
            y_list.append(label)

    print(f"\nDataset: {len(X_list)} samples ({skipped} skipped)")
    if not X_list:
        raise RuntimeError("No training data found. Check VOICE_DIR path and oto.ini files.")

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)

# ─────────────────────────────────────────────
# 5. モデル学習 (scikit-learn)
# ─────────────────────────────────────────────
def train_model(X: np.ndarray, y: np.ndarray):
    """
    MLPRegressorで学習。
    サンプルが少ない場合はRidge回帰にフォールバック。
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split

    print(f"\nTraining with {len(X)} samples, feature dim={X.shape[1]}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    if len(X_train) >= MIN_SAMPLES:
        from sklearn.neural_network import MLPRegressor
        print("Using MLPRegressor (full model)")
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(
                hidden_layer_sizes=(256, 128, 64),
                activation="relu",
                max_iter=500,
                learning_rate_init=0.001,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                verbose=True,
                random_state=42,
            ))
        ])
    else:
        from sklearn.linear_model import Ridge
        print(f"Warning: Only {len(X_train)} samples. Using Ridge regression fallback.")
        print("Collect more UTAU voice libraries for better quality.")
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0))
        ])

    model.fit(X_train, y_train)

    val_score = model.score(X_val, y_val)
    print(f"\nValidation R² score: {val_score:.4f}")
    if val_score < 0.3:
        print("Warning: Low R² score. More diverse training data is recommended.")

    return model

# ─────────────────────────────────────────────
# 6. ONNX エクスポート
# ─────────────────────────────────────────────
def export_onnx(model, output_path: str, input_dim: int):
    """scikit-learnパイプラインをONNXに変換して保存。"""
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        initial_type = [("input", FloatTensorType([None, input_dim]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

        print(f"\nSuccess: ONNX model saved to {output_path}")
        print(f"  Input shape:  (batch, {input_dim})")
        print(f"  Output shape: (batch, 3)  [pre_utterance, overlap, consonant]")

    except ImportError:
        print("\n[Error] skl2onnx not found. Install: pip install skl2onnx")
        print("Saving sklearn model as fallback (model.pkl)...")
        import pickle
        pkl_path = output_path.replace(".onnx", ".pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(model, f)
        print(f"Saved: {pkl_path}")

# ─────────────────────────────────────────────
# 7. VO-SE Pro との接続確認ユーティリティ
# ─────────────────────────────────────────────
def verify_onnx(onnx_path: str, input_dim: int):
    """生成したONNXモデルの動作確認。"""
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path)
        dummy = np.zeros((1, input_dim), dtype=np.float32)
        result = sess.run(None, {"input": dummy})
        print(f"\nVerification OK: output={result[0]}")
        print("  pre_utterance(norm), overlap(norm), consonant(norm)")
        print(f"  → pre_utterance: {result[0][0][0]*300:.1f}ms")
        print(f"  → overlap:       {result[0][0][1]*300:.1f}ms")
        print(f"  → consonant:     {result[0][0][2]*300:.1f}ms")
    except Exception as e:
        print(f"Verification skipped: {e}")

# ─────────────────────────────────────────────
# 8. ダミーデータ生成（音源ゼロ時のテスト用）
# ─────────────────────────────────────────────
def generate_dummy_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """
    実音源がない場合の動作確認用ダミーデータを生成。
    日本語音節の典型的なoto.ini値の統計分布を模倣。
    本番では使わないこと。
    """
    print("\n[DEMO MODE] No voice data found. Generating dummy dataset...")
    print("Place UTAU voice libraries in:", VOICE_DIR)

    rng = np.random.default_rng(42)
    N = 200
    input_dim = N_MFCC * N_FRAMES

    # ランダムなMFCC特徴量
    X = rng.standard_normal((N, input_dim)).astype(np.float32)

    # 日本語音声の典型的なパラメータ分布を模倣
    # pre_utterance: 60-120ms, overlap: 20-60ms, consonant: 30-90ms
    pre_utt  = rng.uniform(60,  120, N) / 300.0
    overlap  = rng.uniform(20,   60, N) / 300.0
    consonant = rng.uniform(30,  90, N) / 300.0
    y = np.stack([pre_utt, overlap, consonant], axis=1).astype(np.float32)

    print(f"Dummy dataset: {N} samples")
    return X, y

# ─────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  VO-SE Pro - AuralAI Model Trainer")
    print("=" * 60)

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
    input_dim   = N_MFCC * N_FRAMES

    # データセット構築
    if os.path.exists(VOICE_DIR) and glob.glob(
        os.path.join(VOICE_DIR, "**", "oto.ini"), recursive=True
    ):
        X, y = build_dataset(VOICE_DIR)
    else:
        # 音源がない場合はダミーデータで動作確認
        X, y = generate_dummy_dataset()

    print(f"\nX shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"y stats (normalized):")
    print(f"  pre_utterance  mean={y[:,0].mean():.3f}  std={y[:,0].std():.3f}")
    print(f"  overlap        mean={y[:,1].mean():.3f}  std={y[:,1].std():.3f}")
    print(f"  consonant      mean={y[:,2].mean():.3f}  std={y[:,2].std():.3f}")

    # 学習
    model = train_model(X, y)

    # ONNXエクスポート
    export_onnx(model, output_path, input_dim)

    # 動作確認
    if os.path.exists(output_path):
        verify_onnx(output_path, input_dim)

    print("\n" + "=" * 60)
    print("  Done! Copy models/aural_dynamics.onnx to your project.")
    print("=" * 60)

if __name__ == "__main__":
    main()
