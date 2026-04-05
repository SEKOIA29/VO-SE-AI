"""
VO-SE Pro - AuralAI Model Trainer (単独音特化版)
単独音UTAU音源から学習データを生成し、
aural_dynamics.onnx を出力するスクリプト。

単独音の特徴:
  - 1WAV = 1oto.iniエントリ（1対1対応）
  - エイリアスが空の場合はWAVファイル名を使う（息継ぎ音など）
  - overlap が小さめ or 0 になりやすい

使い方:
  1. assets/training_voices/ 以下に単独音UTAUフォルダを置く
     (oto.ini と WAVファイルが入っていること)
  2. pip install numpy scipy scikit-learn onnx skl2onnx
     (任意) pip install librosa
  3. python train_aural_model.py
  4. models/aural_dynamics.onnx が生成される
"""

import os
import re
import glob
import wave
import struct
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

# ─────────────────────────────────────────────
# 0. 設定
# ─────────────────────────────────────────────
VOICE_DIR   = "assets/training_voices"
OUTPUT_DIR  = "models"
OUTPUT_NAME = "aural_dynamics.onnx"
TARGET_FS   = 44100
N_MFCC      = 20
N_FRAMES    = 64
MIN_SAMPLES = 10    # 単独音は連続音より音源数が少ないことが多い
MAX_MS      = 300.0 # ラベル正規化の上限(ms)

# 日本語単独音の主要音素リスト（one-hot特徴量用）
# 音源によって異なるが、これをベースに未知音素は <UNK> 扱い
PHONEME_LIST = [
    "あ","い","う","え","お",
    "か","き","く","け","こ",
    "さ","し","す","せ","そ",
    "た","ち","つ","て","と",
    "な","に","ぬ","ね","の",
    "は","ひ","ふ","へ","ほ",
    "ま","み","む","め","も",
    "や","ゆ","よ",
    "ら","り","る","れ","ろ",
    "わ","を","ん",
    "が","ぎ","ぐ","げ","ご",
    "ざ","じ","ず","ぜ","ぞ",
    "だ","ぢ","づ","で","ど",
    "ば","び","ぶ","べ","ぼ",
    "ぱ","ぴ","ぷ","ぺ","ぽ",
    "きゃ","きゅ","きょ",
    "しゃ","しゅ","しょ",
    "ちゃ","ちゅ","ちょ",
    "にゃ","にゅ","にょ",
    "ひゃ","ひゅ","ひょ",
    "みゃ","みゅ","みょ",
    "りゃ","りゅ","りょ",
    "ぎゃ","ぎゅ","ぎょ",
    "じゃ","じゅ","じょ",
    "びゃ","びゅ","びょ",
    "ぴゃ","ぴゅ","ぴょ",
    # ブレス・息継ぎ系（エイリアス空でWAV名使用）
    "息","息2","息3","息4",
    "<UNK>",  # 未知音素
]
PHONEME_INDEX: Dict[str, int] = {p: i for i, p in enumerate(PHONEME_LIST)}
N_PHONEME = len(PHONEME_LIST)


# ─────────────────────────────────────────────
# 1. oto.ini パーサー（単独音特化）
# ─────────────────────────────────────────────
@dataclass
class OtoEntry:
    wav_path:      str
    alias:         str
    offset:        float  # ms
    consonant:     float  # ms
    cutoff:        float  # ms
    pre_utterance: float  # ms
    overlap:       float  # ms


def parse_oto_ini(oto_path: str) -> List[OtoEntry]:
    """
    oto.ini を読み込む。単独音特化の処理:
    - エイリアスが空 → WAVファイル名（拡張子なし）をエイリアスに使う
    - 同じWAVに複数エントリがある場合は最初の1件だけ使う（単独音前提）
    """
    entries: List[OtoEntry] = []
    voice_dir = os.path.dirname(oto_path)
    seen_wavs: set = set()  # [単独音] 1WAV=1エントリを保証

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
            wav_name = wav_name.strip()
            parts = params_str.split(",")
            if len(parts) < 5:
                continue

            # エイリアスが空の場合はWAVファイル名（拡張子なし）を使う
            alias = parts[0].strip()
            if not alias:
                alias = Path(wav_name).stem  # "息.wav" → "息"

            offset    = float(parts[1]) if parts[1].strip() else 0.0
            consonant = float(parts[2]) if parts[2].strip() else 0.0
            cutoff    = float(parts[3]) if parts[3].strip() else 0.0
            pre_utt   = float(parts[4]) if parts[4].strip() else 0.0
            overlap   = float(parts[5]) if len(parts) > 5 and parts[5].strip() else 0.0

            wav_path = os.path.join(voice_dir, wav_name)
            if not wav_path.lower().endswith(".wav"):
                wav_path += ".wav"

            # [単独音] 同じWAVが2度目に出たらスキップ
            if wav_path in seen_wavs:
                continue
            seen_wavs.add(wav_path)

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
# 2. WAV読み込み
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

        if nch == 2:
            data = data[::2]

        if fs != TARGET_FS:
            from math import gcd
            g = gcd(fs, TARGET_FS)
            try:
                from scipy.signal import resample_poly
                data = resample_poly(data, TARGET_FS // g, fs // g).astype(np.float32)
            except ImportError:
                n_new = int(len(data) * TARGET_FS / fs)
                data = np.interp(
                    np.linspace(0, len(data) - 1, n_new),
                    np.arange(len(data)),
                    data
                ).astype(np.float32)

        return data
    except Exception:
        return None


# ─────────────────────────────────────────────
# 3. MFCC抽出
# ─────────────────────────────────────────────
def _stft_numpy(signal: np.ndarray, n_fft=512, hop=256) -> np.ndarray:
    window = np.hanning(n_fft)
    frames = []
    for i in range(0, len(signal) - n_fft, hop):
        frame = signal[i:i + n_fft] * window
        frames.append(np.abs(np.fft.rfft(frame)))
    return np.array(frames).T if frames else np.zeros((n_fft // 2 + 1, 1))


def extract_mfcc_numpy(signal: np.ndarray, n_mfcc=N_MFCC, sr=TARGET_FS) -> np.ndarray:
    n_fft = 512
    hop   = 256
    n_mels = 40
    spec = _stft_numpy(signal, n_fft, hop)
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
    dct_mat  = np.cos(
        np.pi * np.outer(np.arange(n_mfcc), np.arange(1, n_mels + 1) - 0.5) / n_mels
    )
    return np.dot(dct_mat, log_mel)


def extract_mfcc(signal: np.ndarray) -> np.ndarray:
    try:
        import librosa
        return librosa.feature.mfcc(y=signal, sr=TARGET_FS, n_mfcc=N_MFCC)
    except ImportError:
        return extract_mfcc_numpy(signal)


def mfcc_to_fixed_length(mfcc: np.ndarray, n_frames=N_FRAMES) -> np.ndarray:
    if mfcc.shape[1] >= n_frames:
        mfcc = mfcc[:, :n_frames]
    else:
        pad = np.zeros((mfcc.shape[0], n_frames - mfcc.shape[1]))
        mfcc = np.hstack([mfcc, pad])
    return mfcc.flatten().astype(np.float32)


# ─────────────────────────────────────────────
# 4. 音素one-hotベクトル
# ─────────────────────────────────────────────
def phoneme_to_onehot(alias: str) -> np.ndarray:
    """
    音素名をone-hotベクトルに変換。
    未知音素は <UNK> として扱う。

    単独音では音素名が明確なので、MFCCと組み合わせることで
    「同じ声質でも音素によってpreutteranceが違う」を学習できる。
    """
    idx = PHONEME_INDEX.get(alias, PHONEME_INDEX["<UNK>"])
    vec = np.zeros(N_PHONEME, dtype=np.float32)
    vec[idx] = 1.0
    return vec


# ─────────────────────────────────────────────
# 5. 教師データ生成
# ─────────────────────────────────────────────
def build_dataset(voice_root: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    単独音音源から教師データを構築。

    特徴量 X = [MFCC固定長 | 音素one-hot]
      - MFCC: N_MFCC × N_FRAMES = 1280次元（音源の音色・声質）
      - one-hot: N_PHONEME次元（音素の種類）
      → 合計: 1280 + N_PHONEME 次元

    ラベル y = [pre_utterance, overlap, consonant] (0-1正規化)
    """
    oto_files = glob.glob(os.path.join(voice_root, "**", "oto.ini"), recursive=True)
    print(f"Found {len(oto_files)} oto.ini files")

    X_list, y_list = [], []
    skipped_no_wav  = 0
    skipped_too_short = 0

    for oto_path in oto_files:
        entries = parse_oto_ini(oto_path)
        print(f"  {oto_path}: {len(entries)} entries")

        for entry in entries:
            if not os.path.exists(entry.wav_path):
                skipped_no_wav += 1
                continue

            signal = load_wav(entry.wav_path)
            if signal is None:
                skipped_no_wav += 1
                continue

            # 単独音は短いものが多いので閾値を低めに設定（最低0.03秒）
            if len(signal) < TARGET_FS * 0.03:
                skipped_too_short += 1
                continue

            # offset位置から切り出し（offsetより前はブランク）
            start = int(entry.offset * TARGET_FS / 1000)
            start = max(0, min(start, len(signal) - 1))
            segment = signal[start:]

            if len(segment) < 256:
                skipped_too_short += 1
                continue

            # MFCC特徴量
            mfcc = extract_mfcc(segment)
            mfcc_feat = mfcc_to_fixed_length(mfcc)  # (N_MFCC * N_FRAMES,)

            # 音素one-hot特徴量
            phoneme_feat = phoneme_to_onehot(entry.alias)  # (N_PHONEME,)

            # 結合特徴量
            feat = np.concatenate([mfcc_feat, phoneme_feat])  # (1280 + N_PHONEME,)

            # ラベル
            label = np.array([
                np.clip(entry.pre_utterance / MAX_MS, 0.0, 1.0),
                np.clip(entry.overlap        / MAX_MS, 0.0, 1.0),
                np.clip(entry.consonant      / MAX_MS, 0.0, 1.0),
            ], dtype=np.float32)

            X_list.append(feat)
            y_list.append(label)

    print(f"\nDataset summary:")
    print(f"  Valid samples   : {len(X_list)}")
    print(f"  Skipped (no WAV): {skipped_no_wav}")
    print(f"  Skipped (short) : {skipped_too_short}")

    if not X_list:
        raise RuntimeError(
            "No training data found.\n"
            f"Check VOICE_DIR: {voice_root}\n"
            "Make sure oto.ini and WAV files are in the same folder."
        )

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


# ─────────────────────────────────────────────
# 6. モデル学習
# ─────────────────────────────────────────────
def train_model(X: np.ndarray, y: np.ndarray):
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split

    print(f"\nTraining: {len(X)} samples, feature dim={X.shape[1]}")
    print(f"  MFCC部分    : {N_MFCC * N_FRAMES} 次元")
    print(f"  音素one-hot : {N_PHONEME} 次元")

    # サンプル数が少ない場合はtrain/val分割なし
    if len(X) < 20:
        X_train, X_val = X, X
        y_train, y_val = y, y
        print(f"  [注意] サンプルが{len(X)}件と少ないため検証セットなしで学習します")
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.15, random_state=42
        )

    if len(X_train) >= MIN_SAMPLES:
        from sklearn.neural_network import MLPRegressor
        print("  Model: MLPRegressor (256-128-64)")
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(
                hidden_layer_sizes=(256, 128, 64),
                activation="relu",
                max_iter=500,
                learning_rate_init=0.001,
                early_stopping=len(X_train) >= 20,
                validation_fraction=0.1,
                n_iter_no_change=20,
                verbose=True,
                random_state=42,
            ))
        ])
    else:
        from sklearn.linear_model import Ridge
        print(f"  Model: Ridge (サンプル不足のフォールバック)")
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0))
        ])

    model.fit(X_train, y_train)

    val_score = model.score(X_val, y_val)
    print(f"\nValidation R²: {val_score:.4f}")
    if val_score < 0.3:
        print("  [注意] スコアが低いです。音源を増やすと精度が上がります。")

    return model


# ─────────────────────────────────────────────
# 7. ONNX エクスポート
# ─────────────────────────────────────────────
def export_onnx(model, output_path: str, input_dim: int):
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        initial_type = [("input", FloatTensorType([None, input_dim]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

        print(f"\nONNX saved: {output_path}")
        print(f"  input  shape: (batch, {input_dim})")
        print(f"    └ MFCC    : {N_MFCC * N_FRAMES}次元")
        print(f"    └ one-hot : {N_PHONEME}次元")
        print(f"  output shape: (batch, 3)")
        print(f"    └ [pre_utterance, overlap, consonant] (正規化値×{MAX_MS:.0f}=ms)")

    except ImportError:
        print("\n[Error] skl2onnx not found.")
        print("Install: pip install skl2onnx")
        import pickle
        pkl_path = output_path.replace(".onnx", ".pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(model, f)
        print(f"Fallback: saved as {pkl_path}")


# ─────────────────────────────────────────────
# 8. 動作確認
# ─────────────────────────────────────────────
def verify_onnx(onnx_path: str, input_dim: int):
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path)

        # 「あ」のone-hotでテスト
        dummy_mfcc    = np.zeros((1, N_MFCC * N_FRAMES), dtype=np.float32)
        dummy_phoneme = phoneme_to_onehot("あ").reshape(1, -1)
        dummy = np.concatenate([dummy_mfcc, dummy_phoneme], axis=1)

        result = sess.run(None, {"input": dummy})[0]
        print(f"\nVerification OK (音素='あ' のダミー入力):")
        print(f"  pre_utterance: {result[0][0] * MAX_MS:.1f}ms")
        print(f"  overlap      : {result[0][1] * MAX_MS:.1f}ms")
        print(f"  consonant    : {result[0][2] * MAX_MS:.1f}ms")
    except Exception as e:
        print(f"Verification skipped: {e}")


# ─────────────────────────────────────────────
# 9. ダミーデータ（音源ゼロ時のテスト用）
# ─────────────────────────────────────────────
def generate_dummy_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """
    単独音の典型的な値分布を模倣したダミーデータ。
    本番では使わないこと。
    """
    print("\n[DEMO MODE] 音源が見つかりません。ダミーデータで動作確認します。")
    print(f"実際に使う場合は {VOICE_DIR} に単独音UTAU音源を置いてください。")

    rng = np.random.default_rng(42)
    N = 200
    mfcc_dim = N_MFCC * N_FRAMES

    X_list = []
    y_list = []

    # 音素ごとに典型的な値を設定してダミー生成
    phoneme_params = {
        "あ": (80, 40, 30),  # (pre_utt, overlap, consonant) ms
        "い": (75, 35, 28),
        "う": (78, 38, 25),
        "え": (77, 36, 27),
        "お": (82, 42, 32),
        "か": (90, 45, 60),  # 子音があるのでconsonantが長め
        "き": (88, 43, 55),
        "さ": (95, 48, 70),
        "た": (92, 46, 65),
        "な": (85, 42, 45),
    }

    for phoneme, (pre, ov, con) in phoneme_params.items():
        for _ in range(20):
            mfcc_feat    = rng.standard_normal(mfcc_dim).astype(np.float32)
            phoneme_feat = phoneme_to_onehot(phoneme)
            feat = np.concatenate([mfcc_feat, phoneme_feat])

            label = np.array([
                np.clip(rng.normal(pre, 5) / MAX_MS, 0.0, 1.0),
                np.clip(rng.normal(ov,  5) / MAX_MS, 0.0, 1.0),
                np.clip(rng.normal(con, 5) / MAX_MS, 0.0, 1.0),
            ], dtype=np.float32)

            X_list.append(feat)
            y_list.append(label)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    print(f"ダミーデータ: {len(X)} samples")
    return X, y


# ─────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  VO-SE Pro - AuralAI Trainer (単独音特化版)")
    print("=" * 60)

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
    input_dim   = N_MFCC * N_FRAMES + N_PHONEME  # MFCC + one-hot

    # データセット構築
    has_voice = (
        os.path.exists(VOICE_DIR)
        and glob.glob(os.path.join(VOICE_DIR, "**", "oto.ini"), recursive=True)
    )

    if has_voice:
        X, y = build_dataset(VOICE_DIR)
    else:
        X, y = generate_dummy_dataset()

    print(f"\nX shape: {X.shape}  (MFCC {N_MFCC*N_FRAMES}次元 + one-hot {N_PHONEME}次元)")
    print(f"y shape: {y.shape}  [pre_utterance, overlap, consonant]")
    print(f"y stats (×{MAX_MS:.0f}=ms):")
    print(f"  pre_utterance  mean={y[:,0].mean()*MAX_MS:.1f}ms  std={y[:,0].std()*MAX_MS:.1f}ms")
    print(f"  overlap        mean={y[:,1].mean()*MAX_MS:.1f}ms  std={y[:,1].std()*MAX_MS:.1f}ms")
    print(f"  consonant      mean={y[:,2].mean()*MAX_MS:.1f}ms  std={y[:,2].std()*MAX_MS:.1f}ms")

    model = train_model(X, y)
    export_onnx(model, output_path, input_dim)

    if os.path.exists(output_path):
        verify_onnx(output_path, input_dim)

    print("\n" + "=" * 60)
    print("  完了! models/aural_dynamics.onnx をプロジェクトに配置してください。")
    print("=" * 60)


if __name__ == "__main__":
    main()
