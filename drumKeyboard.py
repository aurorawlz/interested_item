import pyaudio
import numpy as np
import time
import sys
import json
import os
import argparse

try:
    from pynput.keyboard import Controller, Key

    KEYBOARD_AVAILABLE = True
    _kb = Controller()
except Exception:
    KEYBOARD_AVAILABLE = False
    _kb = None

# --- 🥁 架子鼓专用参数 ---
# 打击乐瞬态明显，可以调高触发阈值，防止误触发
ONSET_THRESHOLD = 5e8  # 调高：需要明确的打击才能触发
RESET_THRESHOLD = 1e8  # 调高：快速复位，支持连续打击
MIN_ENERGY = 2e7  # 最低能量阈值，过滤轻微触碰

# 打击乐分析窗口更短
CHUNK = 1024  # 减小块大小，提高响应速度
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

# 指纹参数调整
FP_FRAMES = 4  # 减少到4帧，鼓声短不需要太长分析
PEAK_RATIO = 0.7  # 主峰占比，用于区分音色

# 文件路径
MAPPING_FILE = os.path.join(os.path.dirname(__file__), "drum_mapping.json")
FINGERPRINT_FILE = os.path.join(os.path.dirname(__file__), "drum_fingerprints.json")
DRUM_CONFIG_FILE = os.path.join(os.path.dirname(__file__), "drum_config.json")



def load_drum_config():
    """加载架子鼓专用配置"""
    default_config = {
        "sensitivity": {
            "kick": 0.8,  # 地鼓灵敏度
            "snare": 1.0,  # 军鼓灵敏度
            "hihat": 1.2,  # 踩镲灵敏度
            "tom": 0.9,  # 通鼓灵敏度
            "cymbal": 1.3  # 镲片灵敏度
        },
        "velocity_thresholds": {
            "soft": 1e8,  # 轻击阈值
            "medium": 3e8,  # 中击阈值
            "hard": 6e8  # 重击阈值
        }
    }

    if os.path.exists(DRUM_CONFIG_FILE):
        try:
            with open(DRUM_CONFIG_FILE, "r", encoding="utf-8") as f:
                user_config = json.load(f)
                # 合并配置
                for key in default_config:
                    if key in user_config:
                        default_config[key].update(user_config[key])
        except Exception:
            pass

    return default_config


def extract_drum_features(signal):
    """提取打击乐特征"""
    # 1. 能量特征
    energy = np.sum(signal.astype(float) ** 2)

    # 2. 频谱特征
    windowed = signal * np.hanning(len(signal))
    spectrum = np.abs(np.fft.rfft(windowed))

    # 3. 打击乐重要频段
    # 低频段 (50-200Hz): 地鼓
    # 中频段 (200-800Hz): 军鼓
    # 高频段 (2k-8kHz): 踩镲
    freq_bins = len(spectrum)
    freq_resolution = RATE / 2 / freq_bins

    # 计算各频段能量
    low_band = np.sum(spectrum[int(50 / freq_resolution):int(200 / freq_resolution)])
    mid_band = np.sum(spectrum[int(200 / freq_resolution):int(800 / freq_resolution)])
    high_band = np.sum(spectrum[int(2000 / freq_resolution):int(8000 / freq_resolution)])

    total_spectrum = np.sum(spectrum)

    features = {
        "energy": float(energy),
        "low_ratio": float(low_band / total_spectrum if total_spectrum > 0 else 0),
        "mid_ratio": float(mid_band / total_spectrum if total_spectrum > 0 else 0),
        "high_ratio": float(high_band / total_spectrum if total_spectrum > 0 else 0),
        "attack_slope": float(np.max(np.diff(signal[:100])) if len(signal) > 100 else 0),
        "peak_count": int(np.sum(spectrum > np.mean(spectrum) * 2)),
        "centroid": float(np.sum(np.arange(len(spectrum)) * spectrum) / total_spectrum if total_spectrum > 0 else 0)
    }

    # 归一化特征向量用于匹配
    feature_vector = [
        features["low_ratio"],
        features["mid_ratio"],
        features["high_ratio"],
        min(features["attack_slope"] / 1000, 1.0),
        min(features["peak_count"] / 50, 1.0),
        min(features["centroid"] / 1000, 1.0)
    ]

    return features, feature_vector


def classify_drum_type(features, velocity):
    """根据特征初步分类鼓件类型"""
    if features["low_ratio"] > 0.6 and features["attack_slope"] > 500:
        return "kick"  # 地鼓：低频占比高，瞬态强
    elif features["mid_ratio"] > 0.5 and features["peak_count"] > 10:
        return "snare"  # 军鼓：中频丰富，谐波多
    elif features["high_ratio"] > 0.7:
        return "hihat" if velocity < 4e8 else "cymbal"  # 踩镲或镲片
    elif features["mid_ratio"] > 0.4 and features["low_ratio"] > 0.3:
        return "tom"  # 通鼓
    else:
        return "unknown"


def classify_velocity(energy):
    """根据能量分级打击力度"""
    config = load_drum_config()
    thresholds = config["velocity_thresholds"]

    if energy < thresholds["soft"]:
        return "soft"
    elif energy < thresholds["medium"]:
        return "medium"
    else:
        return "hard"


def load_mapping():
    """加载鼓件映射"""
    if not os.path.exists(MAPPING_FILE):
        # 创建默认架子鼓映射
        default_mapping = {
            "kick": ["space", "z", "x"],  # 地鼓：空格/轻/z, 中/x, 重/c
            "snare": ["a", "s", "d"],  # 军鼓
            "hihat": ["q", "w", "e"],  # 踩镲
            "tom_high": ["r", "t", "y"],  # 高音通鼓
            "tom_mid": ["f", "g", "h"],  # 中音通鼓
            "tom_low": ["v", "b", "n"],  # 低音通鼓
            "cymbal": ["u", "i", "o"],  # 吊镲
            "ride": ["j", "k", "l"],  # 叮叮镲
            "crash": ["m", ",", "."]  # 碎音镲
        }
        save_mapping(default_mapping)
        return default_mapping

    try:
        with open(MAPPING_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_mapping(mapping):
    try:
        with open(MAPPING_FILE, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def load_fingerprints():
    if not os.path.exists(FINGERPRINT_FILE):
        return {}
    try:
        with open(FINGERPRINT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_fingerprints(db):
    try:
        with open(FINGERPRINT_FILE, "w", encoding="utf-8") as f:
            json.dump(db, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def _parse_key_name(name: str):
    """键盘映射解析"""
    if not name:
        return None
    lower = name.lower()
    special_map = {
        "space": Key.space,
        "enter": Key.enter,
        "return": Key.enter,
        "tab": Key.tab,
        "esc": Key.esc,
        "escape": Key.esc,
        "backspace": Key.backspace,
        "shift": Key.shift,
        "ctrl": Key.ctrl,
        "control": Key.ctrl,
        "alt": Key.alt,
        "up": Key.up,
        "down": Key.down,
        "left": Key.left,
        "right": Key.right,
    }
    return special_map.get(lower, name)


def send_keypress(key_name: str):
    """发送按键"""
    if not KEYBOARD_AVAILABLE or _kb is None:
        return

    target = _parse_key_name(key_name)
    if target is None:
        return

    if isinstance(target, Key):
        _kb.press(target)
        _kb.release(target)
    elif isinstance(target, str):
        for ch in target:
            _kb.press(ch)
            _kb.release(ch)


def _gather_frames(stream, first_chunk: np.ndarray, frames: int) -> np.ndarray:
    """采集多帧数据"""
    if frames <= 1:
        return first_chunk
    buf = [first_chunk]
    for _ in range(frames - 1):
        raw = stream.read(CHUNK, exception_on_overflow=False)
        arr = np.frombuffer(raw, dtype=np.int16)
        buf.append(arr)
    return np.concatenate(buf)


def cosine_similarity(vec_a, vec_b):
    """余弦相似度"""
    n = min(len(vec_a), len(vec_b))
    if n == 0:
        return 0.0
    a = np.array(vec_a[:n])
    b = np.array(vec_b[:n])
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def draw_drum_ui(energy, velocity_level, drum_type, is_ready):
    """绘制架子鼓专用界面"""
    # 能量条
    if energy < 1: energy = 1
    log_energy = np.log10(energy)
    bar_len = int((log_energy - 7) * 8)  # 调整显示范围
    if bar_len < 0: bar_len = 0
    if bar_len > 50: bar_len = 50

    bar = "█" * bar_len + "░" * (50 - bar_len)

    # 力度指示器
    velocity_icons = {"soft": "○", "medium": "◎", "hard": "●"}
    velocity_icon = velocity_icons.get(velocity_level, "○")

    # 鼓件颜色
    drum_colors = {
        "kick": "🟤", "snare": "🔴", "hihat": "🟢",
        "tom": "🟡", "cymbal": "🔵", "unknown": "⚪"
    }
    drum_icon = drum_colors.get(drum_type, "⚪")

    status = "🟢 READY" if is_ready else "🔴 LOCKED"

    sys.stdout.write(f"\r[{drum_icon}{velocity_icon}] [{bar}] {drum_type:8} {velocity_level:6} | {status}")
    sys.stdout.flush()


def run_drum_detector(use_map=False, record_drum=None, record_key=None,
                      sim_threshold=0.85, build_map=False, auto_classify=False):
    """架子鼓主检测函数"""
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                    input=True, frames_per_buffer=CHUNK)

    print("\n🥁 架子鼓检测模式启动")
    print(f"触发阈值: {ONSET_THRESHOLD:,} | 复位阈值: {RESET_THRESHOLD:,}")
    print(f"块大小: {CHUNK} | 采样率: {RATE}")
    print("-" * 50)

    if auto_classify:
        print("🤖 自动分类模式：系统将尝试自动识别鼓件类型")

    # 加载配置
    drum_config = load_drum_config()
    mapping = load_mapping()
    fingerprints = load_fingerprints()

    prev_energy = 0
    is_ready_to_trigger = True
    last_trigger_time = 0
    min_interval = 0.05  # 最小触发间隔50ms，防止连击

    try:
        while True:
            current_time = time.time()

            raw_data = stream.read(CHUNK, exception_on_overflow=False)
            data_int = np.frombuffer(raw_data, dtype=np.int16)

            curr_energy = np.sum(data_int.astype(float) ** 2)
            flux = curr_energy - prev_energy
            prev_energy = curr_energy

            # 计算力度等级
            velocity_level = classify_velocity(curr_energy)

            # 状态机
            if is_ready_to_trigger:
                # 检测打击触发
                if flux > ONSET_THRESHOLD and curr_energy > MIN_ENERGY:
                    # 检查时间间隔
                    if current_time - last_trigger_time < min_interval:
                        continue

                    last_trigger_time = current_time

                    # 采集音频数据
                    combined = _gather_frames(stream, data_int, FP_FRAMES)

                    # 提取特征
                    features, feature_vector = extract_drum_features(combined)

                    # 自动分类鼓件类型
                    drum_type = classify_drum_type(features, curr_energy)

                    sys.stdout.write("\n")

                    if build_map:
                        # 交互式录制模式
                        print(f"🎯 检测到打击 | 能量: {curr_energy:.2e} | 自动分类: {drum_type}")

                        if auto_classify:
                            # 使用自动分类结果
                            base_name = drum_type
                            idx = 1
                            while f"{base_name}_{idx}" in fingerprints:
                                idx += 1
                            drum_name = f"{base_name}_{idx}"
                            print(f"🤖 自动命名为: {drum_name}")
                        else:
                            # 手动输入名称
                            drum_name = input("请输入鼓件名称 (如 kick_1, snare, hihat): ").strip()
                            if not drum_name:
                                drum_name = f"drum_{len(fingerprints) + 1}"

                        # 获取按键映射
                        if drum_type in mapping:
                            default_keys = mapping[drum_type]
                            velocity_idx = 0 if velocity_level == "soft" else 1 if velocity_level == "medium" else 2
                            default_key = default_keys[velocity_idx]
                            print(f"💡 建议按键: {default_key} (根据力度自动选择)")

                        key_input = input("请输入按键映射 (回车使用建议键): ").strip()
                        if not key_input and 'default_key' in locals():
                            key_input = default_key

                        if key_input:
                            # 保存指纹
                            fingerprints[drum_name] = {
                                "fingerprint": feature_vector,
                                "features": features,
                                "drum_type": drum_type,
                                "velocity": velocity_level,
                                "key": key_input
                            }

                            if save_fingerprints(fingerprints):
                                print(f"✅ 已保存: {drum_name} -> {key_input}")
                            else:
                                print("❌ 保存失败")

                    elif record_drum:
                        # 指定名称录制模式
                        fingerprints[record_drum] = {
                            "fingerprint": feature_vector,
                            "features": features,
                            "drum_type": drum_type,
                            "velocity": velocity_level,
                            "key": record_key or ""
                        }

                        if save_fingerprints(fingerprints):
                            print(f"✅ 已录制: {record_drum} | 类型: {drum_type} | 力度: {velocity_level}")
                        else:
                            print("❌ 录制失败")

                    else:
                        # 识别模式
                        best_match = None
                        best_sim = 0.0

                        for name, data in fingerprints.items():
                            stored_vector = data.get("fingerprint", [])
                            sim = cosine_similarity(feature_vector, stored_vector)

                            # 考虑鼓件类型和力度的权重
                            type_bonus = 0.1 if data.get("drum_type", "") == drum_type else 0
                            velocity_bonus = 0.05 if data.get("velocity", "") == velocity_level else 0
                            sim_adj = sim + type_bonus + velocity_bonus

                            if sim_adj > best_sim:
                                best_sim = sim_adj
                                best_match = name

                        if best_match and best_sim >= sim_threshold:
                            drum_data = fingerprints[best_match]
                            key_to_press = drum_data.get("key", "")

                            print(f"🥁 识别: {best_match} | 类型: {drum_type} | 力度: {velocity_level}")
                            print(f"  相似度: {best_sim:.3f} | 键位: {key_to_press}")

                            if key_to_press and use_map and KEYBOARD_AVAILABLE:
                                try:
                                    send_keypress(key_to_press)
                                    print(f"⌨️ 触发按键: {key_to_press}")
                                except Exception as e:
                                    print(f"⚠️ 按键失败: {e}")
                        else:
                            print(f"❓ 未知打击 | 类型: {drum_type} | 力度: {velocity_level}")
                            print(f"  最高相似度: {best_sim:.3f} (阈值: {sim_threshold})")

                    # 锁定系统，防止连击
                    is_ready_to_trigger = False

            else:
                # 锁定状态，等待复位
                if curr_energy < RESET_THRESHOLD:
                    is_ready_to_trigger = True

            # 更新UI显示
            draw_drum_ui(curr_energy, velocity_level,
                         drum_type if 'drum_type' in locals() else "unknown",
                         is_ready_to_trigger)

    except KeyboardInterrupt:

        print("\n\n⏹️ 停止检测")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


def main():
    parser = argparse.ArgumentParser(description="架子鼓音频识别系统")
    parser.add_argument("--use-map", action="store_true", help="识别时发送键盘输入")
    parser.add_argument("--build-map", action="store_true", help="交互式录制模式")
    parser.add_argument("--record-drum", metavar="NAME", help="录制指定名称的鼓件")
    parser.add_argument("--record-key", metavar="KEY", help="录制时指定按键映射")
    parser.add_argument("--sim-th", type=float, default=0.85, help="相似度阈值，默认0.85")
    parser.add_argument("--auto-classify", action="store_true", help="自动分类鼓件类型")
    parser.add_argument("--list-drums", action="store_true", help="显示已录制的鼓件")

    args = parser.parse_args()

    if args.list_drums:
        fingerprints = load_fingerprints()
        if fingerprints:
            print("\n🥁 已录制的鼓件:")
            print("-" * 60)
            for name, data in fingerprints.items():
                drum_type = data.get("drum_type", "unknown")
                velocity = data.get("velocity", "unknown")
                key = data.get("key", "未设置")
                print(f"{name:20} | 类型: {drum_type:10} | 力度: {velocity:8} | 按键: {key}")
        else:
            print("📭 未找到任何鼓件记录")
        return

    run_drum_detector(
        use_map=args.use_map,
        record_drum=args.record_drum,
        record_key=args.record_key,
        sim_threshold=args.sim_th,
        build_map=args.build_map,
        auto_classify=args.auto_classify
    )


if __name__ == "__main__":
    main()